
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x):
        x1 = self.linear(x)
        x2 = x1 + 3
        x3 = torch.clamp_min(x2, 0)
        x4 = torch.clamp_max(x3, 6)
        x5 = x4 / 6
        return x5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
