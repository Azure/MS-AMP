# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""The fsdp mnist example using MS-AMP.

It is adapted from https://github.com/pytorch/examples/blob/main/mnist/main.py.
"""

import os
import argparse
import functools

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim.lr_scheduler import StepLR
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms


def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group('nccl', rank=rank, world_size=world_size)


def cleanup():
    """Destroy the distributed environment."""
    dist.destroy_process_group()


class Net(nn.Module):
    """The neural network model for mnist."""
    def __init__(self):
        """Constructor."""
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """Forward function."""
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    """Train the model with given data loader and optimizer."""
    model.train()
    ddp_loss = torch.zeros(2).to(rank)
    if sampler:
        sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(rank), target.to(rank)

        optimizer.zero_grad()

        output = model(data)
        loss = F.nll_loss(output, target, reduction='sum')
        loss.backward()

        optimizer.step()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(data)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, ddp_loss[0] / ddp_loss[1]))


def test(model, rank, world_size, test_loader):
    """Test the model on test data set."""

    model.eval()
    ddp_loss = torch.zeros(3).to(rank)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(rank), target.to(rank)
            output = model(data)
            ddp_loss[0] += F.nll_loss(output, target, reduction='sum').item()    # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)    # get the index of the max log-probability
            ddp_loss[1] += pred.eq(target.view_as(pred)).sum().item()
            ddp_loss[2] += len(data)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    if rank == 0:
        test_loss = ddp_loss[0] / ddp_loss[2]
        print(
            'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                test_loss, int(ddp_loss[1]), int(ddp_loss[2]), 100. * ddp_loss[1] / ddp_loss[2]
            )
        )


def fsdp_main(rank, world_size, args):
    """The main function for fsdp mnist example."""
    setup(rank, world_size)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081, ))])

    if rank == 0:
        dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dist.barrier()
    if rank != 0:
        dataset1 = datasets.MNIST('./data', train=True, transform=transform)

    dataset2 = datasets.MNIST('./data', train=False, transform=transform)

    sampler1 = DistributedSampler(dataset1, rank=rank, num_replicas=world_size, shuffle=True)
    sampler2 = DistributedSampler(dataset2, rank=rank, num_replicas=world_size)

    train_kwargs = {'batch_size': args.batch_size, 'sampler': sampler1}
    test_kwargs = {'batch_size': args.test_batch_size, 'sampler': sampler2}
    cuda_kwargs = {'num_workers': 2, 'pin_memory': True, 'shuffle': False}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    my_auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=20000)
    torch.cuda.set_device(rank)

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    model = Net().to(rank)

    if args.msamp:
        from msamp.fsdp import FsdpReplacer
        from msamp.fsdp import FP8FullyShardedDataParallel
        model = FsdpReplacer.replace(model)
        model = FP8FullyShardedDataParallel(model, use_orig_params=True, auto_wrap_policy=my_auto_wrap_policy)
    else:
        model = FSDP(model, use_orig_params=True, auto_wrap_policy=my_auto_wrap_policy)

    if rank == 0:
        print(f'FSDP model: {model}')

    if args.msamp:
        from msamp.optim import FSDPAdam
        optimizer = FSDPAdam(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    init_start_event.record()
    for epoch in range(1, args.epochs + 1):
        train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=sampler1)
        test(model, rank, world_size, test_loader)
        scheduler.step()

    init_end_event.record()

    if args.save_model:
        # use a barrier to make sure training is done on all ranks
        dist.barrier()
        states = model.state_dict()
        if rank == 0:
            torch.save(states, 'mnist_cnn.pt')

    cleanup()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument(
        '--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)'
    )
    parser.add_argument(
        '--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)'
    )
    parser.add_argument('--epochs', type=int, default=14, metavar='N', help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR', help='learning rate (default: 3e-4)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M', help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--msamp', action='store_true', default=False, help='whether use MS-AMP')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(fsdp_main, args=(WORLD_SIZE, args), nprocs=WORLD_SIZE, join=True)
