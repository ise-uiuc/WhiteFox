
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(4, 5)
 
    def forward(self, x1):
        v1 = self.l(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        return v4 / 6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4)
