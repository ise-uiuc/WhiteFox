
import torch
import torch.nn
import torch.nn.functional as F
import warnings
import math
from torch.nn.init import trunc_normal_
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
def fix_ncrops_dims(x):
    if x.dim() == 2:
        x = x.unsqueeze(-1)
    return x
class Split(torch.nn.Module):
    def __init__(self, num_features, num_heads, dim = None):
        super().__init__()
        if dim is None:
            dim = num_features
        self.bias = torch.nn.Parameter(torch.zeros(1, 1, num_heads, 1, num_features))
        self.b = torch.nn.Parameter(torch.zeros(1, 1, num_heads, num_features, 1))
    def forward(self, x):
        x_b = x + self.bias
        x_b = x_b.reshape(x_b.size(0), x_b.size(2), -1)
        x_ = self.b.transpose(-2, -1)
        x = torch.bmm(x_b, x_)
        x = x.reshape(-1, x.size(1))
        x = x.transpose(0, 1)
        return x
class MultiheadAttention(torch.nn.Module):
    def __init__(self, num_heads, hidden_dim = None, dropout = 0.):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = num_heads * 32
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == self.hidden_dim, "hidden_dim must be divisible by num_heads"
        self.in_proj_weight = torch.nn.Parameter(torch.Tensor(3 * hidden_dim, hidden_dim))
        self.qkv_proj_weight = torch.nn.Parameter(torch.Tensor(2 * hidden_dim, hidden_dim))
        self.out_proj_weight = torch.nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.in_proj_bias = torch.nn.Parameter(torch.Tensor(3 * hidden_dim))
        self.qkv_proj_bias = torch.nn.Parameter(torch.Tensor(2 * hidden_dim))
        self.out_proj_bias = torch.nn.Parameter(torch.Tensor(hidden_dim))
        self.split_0 = Split(hidden_dim, num_heads)
        self.split_1 = Split(hidden_dim, num_heads)
        self.split_2 = Split(hidden_dim, num_heads)
        self.add_0 = torch.nn.quantized.FloatFunctional()
        self.add_1 = torch.nn.quantized.FloatFunctional()
        self.normalize = torch.nn.LayerNorm(hidden_dim)
        self.proj_weight = torch.nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.proj_bias = torch.nn.Parameter(torch.Tensor(hidden_dim))
        self.rpe = torch.nn.Sequential()
        self.quant = torch.nn.quantized.FloatFunctional()
        self.project_weights_bias()
    def project_weights_bias(self):
        trunc_normal_(self.in_proj_weight, std = 0.02)
        trunc_normal_(self.qkv_proj_weight, std = 0.02)
        trunc_normal_(self.out_proj_weight, std = 0.02)
        trunc_normal_(self.in_proj_bias, std = 0.02)
        trunc_normal_(self.qkv_proj_bias, std = 0.02)
        trunc_normal_(self.out_proj_bias, std = 0.02)
        trunc_normal_(self.proj_weight, std = 0.02)
        trunc_normal_(self.proj_bias, std = 0.02)
    def forward(self, query, key, value, attn_mask, key_padding_mask):
        query = query if query is not None else torch.empty(query.size(), requires_grad = False)
        key = key if key is not None else torch.empty(key.size(), requires_grad = False)
        value = value if value is not None else torch.empty(value.size(), requires_grad = False)
        attn_mask = attn_mask if attn_mask is not None else  torch.empty(attn_mask.size(), requires_grad = False)
        key_padding_mask = key_padding_mask if key_padding_mask is not None else torch.empty(key_padding_mask.size(), requires_grad = False)
        head_dim = self.head_dim
        num_heads = 1
        proj_query, proj_key, proj_value = torch.quantized.dequantize(F.linear(query, self.qkv_proj_weight, self.qkv_proj_bias), scale = 1.0, zero_point = 0)
        q, k, v = proj_query, proj_key, proj_value
        q = self.split_0(q).contiguous().view(q.size(2), q.size(0), num_heads, head_dim).transpose(0, 1).reshape(-1, q.size(1))
        k = self.split_1(k).contiguous().view(k.size(2), k.size(0), num_heads, head_dim).transpose(0, 1).reshape(-1, k.size(1))
        v = self.split_2(v).contiguous().view(v.size(2), v.size(0), num_heads, head_dim).transpose(0, 1).reshape(-1, v.size(1))
        attn_mask = attn_mask.transpose(0, 1).reshape(1, -1)
        key_padding_mask = key_padding_mask.transpose(0, 1).reshape(1, -1)
        qkv_same = torch.equal(q, k) and torch.equal(k, v)
        kv_same = torch.equal(k, v)
        if self.training:
            if qkv_same:
                warnings.warn("self-attention is not unique")
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        if attn_mask is not None:
            attn_output_weights = attn_output_weights + attn_mask
        # if key_padding_mask is not None:
        #     raise NotImplementedError
        attn_output_weights = attn_output_weights.view(attn_output_weights.size(0), -1)
        proj_shape = (-1, head_dim, k.size(1))
        attn_output_weights = self.quant.dequantize(F.linear(self.quant.quantize_per_tensor(attn_output_weights, 1, 0), self.out_proj_weight.view(*proj_shape), self.out_proj_bias.view(*proj_shape)))
        if qkv_same:
            attn_output = attn_output_weights
            return attn_output
        attn_output_weights = F.softmax(attn_output_weights, dim = 0)
        attn_output_weights = torch.nn.functional.dropout(attn_output_weights, p = self.dropout, training = self.training)
        attn_output_weights = attn_output_weights.view(attn_output_weights.size(0), num_heads, -1).transpose(0, 1)
        attn_output = torch.matmul(attn_output_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(attn_output.size(1), attn_output.size(2))
        attn_output = self.add_1.add_scalar(attn_output, self.quant.quantize_per_tensor(256, 1, 0))
        attn_output = F.linear(attn_output, self.in_proj_weight, self.in_proj_bias)
        attn_output = self.normalize(attn_output)
        proj_shape = attn_output.size(0), attn_output.size(1)
        attn_output = self.quant.dequantize(F.linear(self.quant.quantize_per_tensor(attn_output, 128, 0), self.proj_weight.view(*proj_shape), self.proj_bias.view(*proj_shape)))
        return attn_output
class TransformerEncoderLayer(torch.nn.Module):
    def __init__(self, self_attn, feed_forward, dropout, quant_noise, seq_len):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.quant_noise = quant_noise
        self.seq_len = seq_len
        self.eps = 1e-10
        self.add = torch.nn.quantized.FloatFunctional()
        self.add_scalar = torch.nn.quantized.FloatFunctional()
        self.normalize = torch.nn.LayerNorm(self.self_attn.hidden_dim, eps = 1e-10)
        self.self_attn_layer_norm = torch.nn.LayerNorm(self.self_attn.hidden_dim, eps = 1e-10)
        self.dropout = torch.nn.Dropout(dropout)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.hardswish = torch.nn.Hardswish(inplace = False)
    def forward(self, src, src_mask = None, src_key_padding_mask = None):
        src1, src2, src3 = src, src, src
        src2 = self.self_attn_layer_norm(src2)
        src2 = self.self_attn(src2, src2, src2, attn_mask = src_mask, key_padding_mask = src_key_padding_mask)[0]
        src = self.add.add(src1, src2)
        src = self.normalize(src)
        src = self.dropout(src)
        src2 = src
        src2 = self.self_attn_layer_norm(src2)
        src2 = self.self_attn(src2, src2, src2, attn_mask = src_mask, key_padding_mask = src_key_padding_mask)[0]
        src = self.add.add(src1, src2)
        src = self.normalize(src)
        src = self.dropout(src)
        src2 = src
        src2 = self.self_attn_layer_norm(src2)
        src2 = self.self_attn(src2, src2, src2, attn_mask = src_mask, key_padding_mask = src_key_padding_mask)[0]
        src3 = self.add_scalar.add_scalar(src3, 1.0)
        src2 = self.add.add(src2, src3)
        src = self.add.add(src1, src2)
        src = self.normalize(src)
        src = self.dropout(src)
        src = self.feed_forward(src)
        return src
class TransformerEncoder(torch.nn.Module):
    def __init__(self, layer, num_layers, norm = None):
        super().__init__()
        self.layers = torch.nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
    def forward(self, src, mask = None, src_key_padding_mask = None):
        output = src
        for i, mod in enumerate(self.layers):
            output = mod(output, src_mask = mask, src_key_padding_mask = src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 64, (7, 7), (2, 2), (3, 3), bias = False), torch.nn.BatchNorm2d(64, 1e-05, 0.1, True), torch.nn.ReLU6(inplace = True), torch.nn.MaxPool2d(kernel_size = (3, 3), stride = (2, 2), padding = (1, 1)), torch.nn.Sequential(torch.nn.Sequential(torch.nn.Sequential(torch.nn.Conv2d(64, 64, (3, 3), 1, 1, bias = False), torch.nn.BatchNorm2d(64, 1e-05, 0.1, True), torch.nn.ReLU6(inplace = True)), torch.nn.Sequential(torch.nn.Conv2d(64, 64, (1, 1), 1, 0, bias = False), torch.nn.BatchNorm2d(64, 1e-05, 0.1, True)), torch.nn.Conv2d(64, 256, (1, 1), 1, 0, bias = False), torch.nn.BatchNorm2d(256, 1e-05, 0.1, True), torch.nn.ReLU6(inplace = True))), torch.nn.Sequential(torch.nn.Sequential(torch.nn.Sequential(torch.nn.Conv2d(256, 64, (1, 1), 1, 0, bias = False), torch.nn.BatchNorm2d(64, 1e-05, 0.1, True), torch.nn.ReLU6(inplace = True)), torch.nn.Sequential(torch.nn.Conv2d(64, 64, (3, 3), 1, 1, bias = False), torch.nn.BatchNorm2d(64, 1e-05, 0.1, True), torch.nn.ReLU6(inplace = True)), torch.nn.Sequential(torch.nn.Conv2d(64, 64, (1, 1), 1, 0, bias = False), torch.nn.BatchNorm2d(64, 1e-05, 0.1, True)), torch.nn.Conv2d(64, 256, (1, 1), 1, 0, bias = False), torch.nn.BatchNorm2d(2