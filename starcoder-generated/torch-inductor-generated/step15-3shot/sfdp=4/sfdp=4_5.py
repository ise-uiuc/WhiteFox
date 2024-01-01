
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        v1 = x1 @ x2.transpose(-2, -1)
        v1 = v1 / math.sqrt(v1.size(-1))
        v1 = v1 + x3
        v2 = torch.softmax(v1, dim=-1)
        v3 = v2 @ x3
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 3, 8)
x2 = torch.randn(2, 8, 4)
x3 = torch.randn(2, 4, 7)
