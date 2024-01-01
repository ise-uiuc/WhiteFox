
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(32, 4)
 
    def forward(self, q1, k1, v1, mask=None):
        o1 = self.attn(q1, k1, v1, mask)
        return o1[0]

# Initializing two tensors
q1 = torch.randn(2, 20, 32)
k1 = torch.randn(2, 10, 32)
v1 = torch.randn(2, 10, 32)

# Initializing the attention mask
mask = torch.randn(2, 20, 10).to(torch.bool)

# Inputs to the model
