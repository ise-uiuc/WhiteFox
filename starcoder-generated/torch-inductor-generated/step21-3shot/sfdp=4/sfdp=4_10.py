
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Linear(32, 512)
 
    def forward(self, x1, x2):
        v1 = self.w(x1)
        v1 = v1.transpose(-2, -1)
 
        v2 = self.w(x2)
        v2 = v2.transpose(-2, -1)
 
        v3 = (v1 @ v2) / (v1.size(-1)**0.5) # scaled dot-product attention
 
        attn_weight = torch.ones_like(v3)
        attn_weight[15:, -3:] = 0
        v4 = (attn_weight @ v3).transpose(-2, -1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16, 32)
x2 = torch.randn(1, 16, 32)
