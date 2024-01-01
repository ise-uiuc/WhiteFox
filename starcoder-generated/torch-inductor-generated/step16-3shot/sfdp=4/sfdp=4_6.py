
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        v1 = x1 @ x2.transpose(-2, -1)
        v1 = v1 / math.sqrt(v1.size(-1))
        v1 = v1 + (torch.rand(v1.shape) + 1)
        v2 = torch.softmax(v1, dim=-1)
        v3 = v2 @ x2
        return v3

# Initializing the model
m = Model()

# Inputs to the model
d1 = 10
d2 = 6
x1 = torch.randn(1, d1, 8)
x2 = torch.randn(1, d2, 8)
