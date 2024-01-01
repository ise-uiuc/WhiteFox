
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
 
    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        
        qk = torch.matmul(query, key.transpose(-2, -1)) 
        scaled_qk = qk.div(inv_scale_factor) 
        softmax_qk = scaled_qk.softmax(dim=-1) 
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p) 
        output = dropout_qk.matmul(value) 
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 8, 64, 64)
key = torch.randn(1, 8, 64, 64)
value = torch.randn(1, 8, 64, 64)
inv_scale_factor = torch.rand((1, 1, 1)).fill_(1e-8).requires_grad_(False)
dropout_p = 0.5
__output = m(query, key, value, inv_scale_factor, dropout_p)

# Test case
import itertools
import torch
from torch import nn
import torch.quantization
import numpy as np
import unittest

from Model import Model

from test.mlir.testing_utils import DisconnectedTestCase

# Targeted at NNC
class ModelTest(DisconnectedTestCase):
    def runTest(self):       
        m = Model()
        m.eval()
        mp = torch.quantization.quantize_dynamic(
            m, {torch.nn.Linear}, dtype=torch.qint8
        )
        mq = torch.quantization.quantize_dynamic(
            mp, {torch.nn.Conv2d}, dtype=torch.qint8
        )

        with torch.no_grad():
            # For torch.quantsim.model_runner
            def gen_input():
                x1 = torch.rand(1, 3, 64, 64, dtype=torch.float)
                __output = m(x1)
                return [np.float32(x1)]

            def is_equal(a, b):
                return np.allclose(a.numpy(), b.numpy(), atol=3e-4)

            yield self.assert_opset_is(gen_input, mq, is_equal)

            # For torch.quantization.fuse_modules
            query = torch.randn(1, 8, 64, 64, dtype=torch.float)
            key = torch.randn(1, 8, 64, 64, dtype=torch.float)
            value = torch.randn(1, 8, 64, 64, dtype=torch.float)
            inv_scale_factor = torch.tensor([[[[1e-8]]]], dtype=torch.float)
            dropout_p = 0.5
 
            def gen_input_fuse():
                __output = m(query, key, value, inv_scale_factor, dropout_p)
                return [np.float32(query), np.float32(key), np.float32(value), np.float32(inv_scale_factor), np.float32(dropout_p)]
            
            yield self.assert_opset_is(gen_input_fuse, mq, is_equal)

            # For torch.quantization.fuse_modules
            query = torch.randn(1, 16, 64, 64, dtype=torch.float)
            key = torch.randn(1, 16, 64, 64, dtype=torch.float)
            value = torch.randn(1, 16, 64, 64, dtype=torch.float)
            inv_scale_factor = torch.tensor([[[[1e-8]]]], dtype=torch.float)
            dropout_p = 0.5

            def gen_input_fuse():
                __output = m(query, key, value, inv_scale_factor, dropout_p)
                return [np.float32(query), np.float32(key), np.float32(value), np.float32(inv_scale_factor), np.float32(dropout_p)]
 
            yield self.assert_opset_is(gen_input_fuse, mq, is_equal)
            
            # For torch.quantization.quantization_mappings
            def gen_input_qsim():
                query = torch.randn(8, 3, 32, 32, dtype=torch.float)
                key = torch.randn(8, 3, 32, 32, dtype=torch.float)
                value = torch.randn(8, 3, 32, 32, dtype=torch.float)

                m = Model()
                mp = torch.quantization.quantize_dynamic(
                    m, {torch.nn.Linear}, dtype=torch.qint8
                )
                mq = torch.quantization.quantize_dynamic(
                    mp, {torch.nn.Conv2d}, dtype=torch.qint8
                )

                mp = torch.quantization.quantize_dynamic(
                    mq, {torch.nn.Conv2d}, dtype=torch.float
                )
                mq2 = torch.quantization.quantize_dynamic(mp, {torch.nn.Conv2d})

                mp = torch.quantization.quantize_dynamic(
                    mq2, {torch.nn.Conv2d}, dtype=torch.qint8
                )
                mq3 = torch.quantization.quantize_dynamic(mp, {torch.nn.Conv2d})
                
                __output = mq3(query)
                return [np.float32(query), np.float32(key), np.float32(value)]
                
            yield self.assert_opset_is(gen_input_qsim, mq, is_equal)

            # For torch.quantization.quantize_fx
            def gen_input_qfx():
                query = torch.randn(8, 3, 32, 32, dtype=torch.float)
                key = torch.randn(8, 3, 32, 32, dtype=torch.float)
                value = torch.randn(8, 3, 32, 32, dtype=torch.float)

                m = Model()
                mp = torch.quantization.quantize_dynamic(
                    m, {torch.nn.Linear}, dtype=torch.qint8
                )
                mq = torch.quantization.quantize_dynamic(
                    mp, {torch.nn.Conv2d}, dtype=torch.qint8
                )

                __output = mq(query)
                return [np.float32(query), np.float32(key), np.float32(value)]

            yield self.assert_opset_is(gen_input_qfx, mq, is_equal)
