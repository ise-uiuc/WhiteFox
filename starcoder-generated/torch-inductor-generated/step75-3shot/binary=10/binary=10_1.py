
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x1, other):
        v1 = x1.view(x1.size(0), -1)
        v2 = torch.matmul(v1, other)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10, 8)
x2 = torch.randn(2)
