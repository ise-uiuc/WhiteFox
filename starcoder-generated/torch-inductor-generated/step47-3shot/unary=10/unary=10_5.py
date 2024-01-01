
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 18)
 
    def forward(self, x1):
        y = self.linear(x1)
        z = y + 3
        z = torch.clamp_min(z, 0)
        z = torch.clamp_max(z, 6)
        z = z / 6
        return z

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
