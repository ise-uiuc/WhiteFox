
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 6)
 
    def forward(self, x):
        y = self.linear(x)
        z = y * torch.clamp(torch.min(y + 3), min=0, max=6)
        w = z / 6
        return w

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3)
