
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = 1.0 - x1.abs()
        v3 = self.linear(v2) - 0.1
        v4 = torch.where(x1 > 0, x1, v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
