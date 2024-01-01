
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(query, key, num_heads, dropout=0.0, bias=True, add_bias_kv=True, add_zero_attn=True)
 
    def forward(self, x1, x2):
        v1, v2 = self.attn(x1, x2, x2)
        return v1, v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64, 64)
x2 = torch.randn(1, 128, 64)
__output1__, __output2__ = m(x1, x2)

