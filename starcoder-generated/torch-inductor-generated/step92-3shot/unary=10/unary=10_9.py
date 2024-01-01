
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)
 
    def forward(self, x1):
        f1 = self.linear(x1)
        f2 = f1 + 3
        f3 = torch.clamp_min(f2, 0)
        f4 = torch.clamp_max(f3, 6)
        f5 = f4 / 6
        return f5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
