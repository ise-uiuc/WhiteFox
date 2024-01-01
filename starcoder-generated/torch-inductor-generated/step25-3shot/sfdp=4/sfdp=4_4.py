
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(8, 8)
 
    def forward(self, x1, x2):
        v1, v2 = self.attn(x1, x2, x2)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32, 8)
x2 = torch.randn(1, 16, 8)
x4 = torch.randn(32, 16)
x5 = torch.randn(1, 16, 8)
__x3__ = self.attention_mask(x5.shape, x4, x5)
