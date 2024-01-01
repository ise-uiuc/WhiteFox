
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(24, 32)
 
    def forward(self, x1, x2):
        v1 = torch.sqrt(x2)
        v2 = self.linear(x2)
        v3 = v1 + v2 - x1
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3)
