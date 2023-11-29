# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for linear module in MS-AMP."""

import io
import copy
import unittest
import torch

from msamp.common.dtype import Dtypes
from msamp.common.tensor import ScalingTensor
from msamp.nn import LinearReplacer
from tests.helper import decorator


class LinearTestCase(unittest.TestCase):
    """Test functions in FP8LInear and LinearReplacer."""
    def setUp(self):
        """Hook method for setting up the test fixture before exercising it."""
        torch.manual_seed(1000)

    def tearDown(self) -> None:
        """Hook method for deconstructing the test fixture after testing it."""
        pass

    @decorator.cuda_test
    def test_fp8linear_forward(self):
        """Test FP8LInear forward function."""
        input = torch.randn((4, 4), device='cuda')
        linear = torch.nn.Linear(4, 8).cuda()
        for qtype in [Dtypes.kfloat32, Dtypes.kfloat16, Dtypes.kbfloat16]:
            model = LinearReplacer.replace(linear, qtype)

            output = linear(input)
            fp8_output = model(input)
            self.assertTrue(fp8_output.dtype == torch.float32)
            self.assertTrue(fp8_output.size() == torch.Size((4, 8)))
            self.assertTrue(torch.allclose(output, fp8_output, 0, 0.1))

    @decorator.cuda_test
    def test_fp8linear_backward(self):
        """Test FP8Linear backward function."""
        input = torch.randn((4, 4), device='cuda')
        linear = torch.nn.Linear(4, 8).cuda()
        linear_copy = copy.deepcopy(linear)

        linear(input).sum().backward()

        for qtype in [Dtypes.kfloat32, Dtypes.kfloat16, Dtypes.kbfloat16]:
            fp8linear = LinearReplacer.replace(linear_copy, qtype)
            fp8linear(input).sum().backward()

            # check bias.
            self.assertTrue(isinstance(fp8linear.bias.grad, torch.Tensor))
            self.assertTrue(torch.equal(fp8linear.bias.grad, linear.bias.grad))

            # check weight.
            self.assertTrue(isinstance(fp8linear.weight.grad, ScalingTensor))
            self.assertTrue(fp8linear.weight.grad.size() == linear.weight.grad.size())

    @decorator.cuda_test
    def test_fp8linear_accu_grad(self):
        """Test accumulate gradient in FP8Linear."""
        input = torch.randn((4, 4), device='cuda')
        linear = torch.nn.Linear(4, 4).cuda()

        model1 = copy.deepcopy(linear)
        model1 = LinearReplacer.replace(model1, Dtypes.kfloat16)
        output1 = model1(input)
        output1.sum().backward()

        model2 = copy.deepcopy(linear)
        model2 = LinearReplacer.replace(model2, Dtypes.kfloat16)
        for i in range(len(input)):
            input2 = input[i:i + 1]
            output2 = model2(input2)
            output2.sum().backward()
        self.assertTrue(torch.allclose(model1.weight.grad.float(), model2.weight.grad.float(), 0, 0.1))

    @decorator.cuda_test
    def test_fp8linear_parameters(self):
        """Test model's parameters of FP8Linear."""
        linear = torch.nn.Linear(4, 8).cuda()
        model = LinearReplacer.replace(linear, Dtypes.kfloat16)
        parameters = dict()
        for name, param in model.named_parameters():
            parameters[name] = param

        self.assertEqual(parameters['weight']._param_name, 'weight')
        self.assertTrue('weight' in parameters)
        self.assertTrue('bias' in parameters)
        self.assertTrue(isinstance(model.weight, ScalingTensor))
        self.assertTrue(torch.allclose(model.weight.float(), linear.weight, rtol=2e-4, atol=1e-3))
        self.assertTrue((linear.bias == model.bias).all())

    @decorator.cuda_test
    def test_meta_mem_immutable(self):
        """Test if meta memory is immutable in FP8Linear."""
        input = torch.randn(4, 4, device='cuda')
        linear = torch.nn.Linear(4, 4).cuda()
        model = LinearReplacer.replace(linear, Dtypes.kfloat16)
        model(input)
        # change mem
        amax = model.scaling_metas['input'].amax
        amax.data = amax.new_zeros(amax.shape)
        with self.assertRaises(RuntimeError):
            model(input)

    @decorator.cuda_test
    def test_linear_output_dtype(self):
        """Test output dtype of FP8Linear."""
        input = torch.randn(4, 4, device='cuda')
        linear = torch.nn.Linear(4, 8).cuda()
        model = LinearReplacer.replace(linear, Dtypes.kfloat16)
        self.assertEqual(model(input).dtype, torch.float32)
        with torch.cuda.amp.autocast():
            self.assertEqual(model(input).dtype, torch.float16)
            self.assertEqual(model(input.half()).dtype, torch.float16)
        model.half()
        self.assertEqual(model(input.half()).dtype, torch.float16)

    @decorator.cuda_test
    def test_linear_custom_attrs(self):
        """Test custom attrs of FP8Linear."""
        linear = torch.nn.Linear(4, 8).cuda()
        linear_attr_abc = 123
        weight_attr_abc = 42
        bias_attr_abc = 100
        linear.abc = linear_attr_abc
        linear.weight.abc = weight_attr_abc
        linear.bias.abc = bias_attr_abc
        model = LinearReplacer.replace(linear, Dtypes.kfloat16)
        # model
        self.assertFalse(model is linear)
        self.assertTrue(hasattr(model, 'abc'))
        self.assertEqual(model.abc, linear_attr_abc)
        # model.weight
        self.assertTrue(hasattr(model.weight, 'abc'))
        self.assertEqual(model.weight.abc, weight_attr_abc)
        # model.bias
        self.assertTrue(hasattr(model.bias, 'abc'))
        self.assertEqual(model.bias.abc, bias_attr_abc)

    @decorator.cuda_test
    def test_state_dict(self):
        """Test state dict of FP8Linear."""
        input = torch.randn((4, 4), device='cuda')
        linear = torch.nn.Linear(4, 8).cuda()
        model1 = LinearReplacer.replace(linear, Dtypes.kfloat16)

        state_dict = model1.state_dict()
        stream = io.BytesIO()
        torch.save(state_dict, stream)
        stream.seek(0)

        model2 = LinearReplacer.replace(linear, Dtypes.kfloat16)
        state_dict = torch.load(stream)
        model2.load_state_dict(state_dict)
        output1 = model1(input)
        output2 = model2(input)
        self.assertTrue(torch.equal(output1, output2))
