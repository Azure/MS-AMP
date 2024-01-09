# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""The deepspeed cifar10 example using MS-AMP and TransformerEngine. It is adapted from official deepspeed example.

The model is adapted from VisionTransfomrer in timm and it uses te.TransformerLayer as encoder block.
"""

import argparse
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import torchvision.transforms as transforms
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import PatchEmbed
from timm.models.vision_transformer_hybrid import HybridEmbed

from msamp import deepspeed


def add_argument():
    """Add arguments."""
    parser = argparse.ArgumentParser(description='CIFAR')

    # data
    # cuda
    parser.add_argument(
        '--with_cuda', default=False, action='store_true', help="use CPU in case there\'s no GPU support"
    )
    parser.add_argument('--use_ema', default=False, action='store_true', help='whether use exponential moving average')

    # train
    parser.add_argument('-b', '--batch_size', default=32, type=int, help='mini-batch size (default: 32)')
    parser.add_argument('-e', '--epochs', default=30, type=int, help='number of total epochs (default: 30)')
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')

    parser.add_argument('--log-interval', type=int, default=200, help='output logging information at a given interval')

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    return args


deepspeed.init_distributed()

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].
# .. note::
#     If running on Windows and you get a BrokenPipeError, try setting
#     the num_worker of torch.utils.data.DataLoader() to 0.

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

if torch.distributed.get_rank() != 0:
    # might be downloading cifar data, let rank 0 download first
    torch.distributed.barrier()

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

if torch.distributed.get_rank() == 0:
    # cifar data is downloaded, indicate other ranks can proceed
    torch.distributed.barrier()

trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

########################################################################
# Let us show some of the training images, for fun.

# functions to show an image


def imshow(img):
    """Show image."""
    img = img / 2 + 0.5    # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

########################################################################
# 2. Define a Convolutional Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).

args = add_argument()


class FP8Block(nn.Module):
    """Transformer Block using TransformerEngine."""
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm
    ):
        """Constructor."""
        super().__init__()
        assert dim % 16 == 0
        mlp_hidden_dim = int(dim * mlp_ratio)
        init_method = partial(trunc_normal_, std=.02)
        self.m = te.TransformerLayer(
            dim,
            mlp_hidden_dim,
            num_heads,
            hidden_dropout=drop,
            attention_dropout=attn_drop,
            self_attn_mask_type='padding',
            layer_type='encoder',
            init_method=init_method,
            output_layer_init_method=init_method,
            drop_path_rate=drop_path,
            fuse_qkv_params=True,
        )

    def forward(self, x):
        """Forward computation."""
        _, batch_size, _ = x.shape
        padding = batch_size % 16 > 0
        if padding:
            x = F.pad(x, (0, 0, 0, 16 - batch_size % 16))
        out = self.m(x, attention_mask=None)
        if padding:
            out = out[:, :batch_size]
        return out


class FP8VisionTransformer(nn.Module):
    """Vision Transformer using TransformerEngine."""
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        hybrid_backbone=None,
        norm_layer=nn.LayerNorm,
        use_checkpoint=False
    ):
        """Constructor."""
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim    # num_features for consistency with other models

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
            )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]    # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                FP8Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer
                ) for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        # self.repr = nn.Linear(embed_dim, representation_size)
        # self.repr_act = nn.Tanh()

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        self.use_checkpoint = use_checkpoint

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        """Forward features."""
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)    # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        # x: (B, L C)

        # (L, B, C)
        x = x.transpose(0, 1).contiguous()
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        x = self.norm(x)
        return x[0]

    def forward(self, x):
        """Forward computation."""
        x = self.forward_features(x)
        x = self.head(x)
        return x


# net = timm.models.vision_transformer.VisionTransformer(32, 2, 3, 10, 32, 4, 2, 4)
net = FP8VisionTransformer(32, 2, 3, 10, 32, 4, 2, 4)

# Initialize DeepSpeed to use the following features
# 1) Distributed model
# 2) Distributed data loader
# 3) DeepSpeed optimizer
model_engine, optimizer, trainloader, __ = deepspeed.initialize(args=args, model=net, training_data=trainset)

print(f'model: {model_engine.module}')
fp16 = model_engine.fp16_enabled()
print(f'fp16={fp16}')

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# net.to(device)
########################################################################
# 3. Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's use a Classification Cross-Entropy loss and SGD with momentum.

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

########################################################################
# 4. Train the network
# ^^^^^^^^^^^^^^^^^^^^
#
# This is when things start to get interesting.
# We simply have to loop over our data iterator, and feed the inputs to the
# network and optimize.

fp8_recipe = recipe.DelayedScaling(margin=0, interval=1, fp8_format=recipe.Format.HYBRID)

for epoch in range(2):    # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(model_engine.local_rank), data[1].to(model_engine.local_rank)
        if fp16:
            inputs = inputs.half()

        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            outputs = model_engine(inputs)
        loss = criterion(outputs, labels)

        model_engine.backward(loss)
        model_engine.step()

        # print statistics
        running_loss += loss.item()
        if i % args.log_interval == (args.log_interval - 1):    # print every log_interval mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / args.log_interval))
            running_loss = 0.0

print('Finished Training')

########################################################################
# 5. Test the network on the test data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We have trained the network for 2 passes over the training dataset.
# But we need to check if the network has learnt anything at all.
#
# We will check this by predicting the class label that the neural network
# outputs, and checking it against the ground-truth. If the prediction is
# correct, we add the sample to the list of correct predictions.
#
# Okay, first step. Let us display an image from the test set to get familiar.

dataiter = iter(testloader)
images, labels = next(dataiter)

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

########################################################################
# Okay, now let us see what the neural network thinks these examples above are:
if fp16:
    images = images.half()
outputs = net(images.to(model_engine.local_rank))

########################################################################
# The outputs are energies for the 10 classes.
# The higher the energy for a class, the more the network
# thinks that the image is of the particular class.
# So, let's get the index of the highest energy:
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

########################################################################
# The results seem pretty good.
#
# Let us look at how the network performs on the whole dataset.

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        if fp16:
            images = images.half()
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            outputs = net(images.to(model_engine.local_rank))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.to(model_engine.local_rank)).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

########################################################################
# That looks way better than chance, which is 10% accuracy (randomly picking
# a class out of 10 classes).
# Seems like the network learnt something.
#
# Hmmm, what are the classes that performed well, and the classes that did
# not perform well:

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        if fp16:
            images = images.half()
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            outputs = net(images.to(model_engine.local_rank))
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels.to(model_engine.local_rank)).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
