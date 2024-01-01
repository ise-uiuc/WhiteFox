
class Model(torch.nn.Module):
    def __init__(self, num_heads=8, hidden_size=768, qkv_bias=False, dropout=0.1, attention_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        self.qkv_linear = torch.nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self._dropout = torch.nn.Dropout(dropout)
        self._attention_dropout = torch.nn.Dropout(attention_dropout)
        self.out_linear = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, x, attn_mask=None):
        original_shape = x.size()

        qkv = self.qkv_linear(x).reshape(original_shape + (3, self.num_heads, -1))
        qkv = qkv.permute((0, 2, 1, 3))

        q, k, v = qkv[0], qkv[1], qkv[2]

        scale_factor = float(self.hidden_size // self.num_heads) ** (-0.5)
     
        q = q.mul_(scale_factor)
        q = torch.matmul(q, k.transpose(1, 2))
        q = self._attention_dropout(torch.softmax(q, dim=-1))
 
        v = torch.matmul(q, v)
        v = self.out_linear(v)

        v = v.permute(2, 0, 1, 3)
        return v.reshape(original_shape)
    

# Initializing 2 models with different settings
m = Model(1, 1)
m2 = Model(2, 1)

# Inputs to the model
x = torch.randn(1, 1, 16, 16)
