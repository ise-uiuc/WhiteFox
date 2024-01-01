
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        v1 = torch.matmul(x1, x2.transpose(-2, -1)) * math.sqrt(x1.size(-1))
        v2 = v1 + x3
        v3 = torch.softmax(v2, dim=-1)
        return torch.matmul(v3, x3)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(5, 6, 3, 2)
x2 = torch.randn(5, 6, 3, 4)
x3 = torch.randn(5, 6, 4, 5)
