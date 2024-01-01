
class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        self.head_dim = model_dim // num_heads
        self.num_heads = num_heads
        w_linear = lambda d, s, i: nn.Linear(d, s, bias=False)
        self.query = w_linear(model_dim, model_dim, "query")
        self.key = w_linear(model_dim, model_dim, "key")
        self.value = w_linear(model_dim, model_dim, "value")
        self.out = w_linear(model_dim, model_dim, "output")
 
    def _scaled_dot_product(self, query, key):
        return torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
 
    def forward(self, x, mask=None, attn_mask=None):
        b = x.shape[0]
        residual, batch_size, num_heads, head_dim = x, x.shape[0], self.num_heads, self.head_dim
        input_x = torch.cat([self.query(x).reshape(b, num_heads, -1), self.key(x).reshape(b, num_heads, -1),
                            self.value(x).reshape(b, num_heads, -1)], dim=-1)
        input_x = input_x.reshape((batch_size * num_heads, -1, head_dim))
 
        # query * key / scale
        qk = self._scaled_dot_product(input_x[:, :, :], input_x)
        qk = qk.reshape((batch_size, num_heads, -1, input_x.size(-1)))
 
        # softmax
        qk_softmax = torch.softmax(qk, dim=-1)
        qk_softmax = torch.dropout(qk_softmax, 0.2, train=True)
 
        # query * value
        output = torch.matmul(qk, input_x)
 
        # combine heads
        if len(output.shape) == 3:
            output = output.permute(0, 2, 1, 3).contiguous()
            output = output.reshape((batch_size, -1, head_dim * num_heads))
        else:
            output = output.reshape((batch_size, -1, head_dim * num_heads))
 
        # final linear
        output = self.out(output)
        output = output + residual
        return output
 
 
class Model(torch.nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        self.mha = MultiHeadAttention(model_dim, num_heads)
 
    def forward(self, x, mask, attn_mask):
        return self.mha(x, mask, attn_mask)

# Initializing the model
m = Model(model_dim=64, num_heads=8)

# Inputs to the model
x = torch.randn(16, 32, 64)
mask = torch.arange(0, 32).expand(16, 32)
attn_mask = 0
