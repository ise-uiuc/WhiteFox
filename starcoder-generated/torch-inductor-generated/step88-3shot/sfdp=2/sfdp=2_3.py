
class Transformer(torch.nn.Module):
    def __init__(self, num_heads, hidden, attn_drop_p):
        super().__init__()
        self.h = num_heads
        self.d_k = hidden // num_heads
        self.linear_q = torch.nn.Linear(hidden, hidden, bias=False)
        self.linear_k = torch.nn.Linear(hidden, hidden, bias=False)
        self.linear_v = torch.nn.Linear(hidden, hidden, bias=False)
        self.linear_o = torch.nn.Linear(hidden, hidden, bias=False)
        self.dropout = torch.nn.Dropout(attn_drop_p)
   
    def forward(self, q, k, v, mask=None):
        B, N, E = q.shape
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)
        atted_outputs = self.scaled_dot_product(q, k, v, mask=mask)
        atted_output = self._combine_heads(atted_outputs)
        scale_factor = torch.rsqrt(torch.tensor(self.d_k).float())
        atted_output = atted_output*scale_factor
        atted_output = self.linear_o(atted_output)
        return atted_output

    def scaled_dot_product(self, q, k, v, mask=None):
        atted_outputs = torch.matmul(q, k.transpose(-2, -1))
        if mask is not None:
            atted_outputs = atted_outputs.masked_fill(mask, -1e9)
        atted_output = torch.softmax(atted_outputs/math.sqrt(self.d_k), dim=-1)
        atted_output = self._dropout_atted(atted_output)
        atted_output = torch.matmul(atted_output, v)
        return atted_output

    def _split_heads(self, x):
        batch_size = x.shape[0]
        head_size = x.shape[1] // self.h
        return x.view(batch_size, self.h, head_size, self.d_k).transpose(1, 2)

    def _dropout_atted(self, x):
        return self.dropout(x)

    def _combine_heads(self, x):
        return x.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], -1)

# Initializing the model
num_heads = 16
hidden = 256
attn_drop_p = 0.2
m = Transformer(num_heads=num_heads, hidden=hidden, attn_drop_p=attn_drop_p)

# Inputs to the model
q = torch.randn(1, 64, 256)
k = torch.randn(1, 256, 256)
v = torch.randn(1, 256, 256)
