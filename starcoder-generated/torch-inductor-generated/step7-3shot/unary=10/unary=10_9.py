
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 6)
 
    def forward(self, x1):
        x2 = self.linear(x1)
        x3 = x2 + 3
        x4 = torch.clamp(x3, 0, 6)
        x5 = x4 / 6
        return x5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
