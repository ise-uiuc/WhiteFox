
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x1, x2):
        x3 = torch.matmul(x1, x2.transpose(-2, -1))
        x4 = x3 / 3.0
        x5 = torch.softmax(x4, dim=-1)
        x6 = torch.matmul(x5, x3)
        return x6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 2, 4, 64, 64)
