
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 64)
 
    def forward(self, x1, x2, x3):
        v1 = self.linear(x1)
        v2 = self.linear(x2)
        v3 = x3.transpose(-2, -1)
        v4 = v2 * v3
        v5 = v1 * v4
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 10)
x2 = torch.randn(7, 8)
x3 = torch.randn(10, 8, 32)
