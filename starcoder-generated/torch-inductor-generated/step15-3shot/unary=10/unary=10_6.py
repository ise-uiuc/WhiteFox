
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(10, 16)
 
    def forward(self, x):
        v = self.l(x)
        v = v + 3
        v = torch.clamp_min(v, 0)
        v = torch.clamp_max(v, 6)
        v = v / 6
        return v

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(5, 10)
