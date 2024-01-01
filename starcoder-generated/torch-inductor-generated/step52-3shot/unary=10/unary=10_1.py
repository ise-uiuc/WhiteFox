
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 1)
 
    def forward(self, x):
        y = self.linear(x)
        z = y + 3
        w1 = torch.clamp_min(z, 0)
        w = torch.clamp_max(w1, 6)
        v = w / 6
        return v

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
