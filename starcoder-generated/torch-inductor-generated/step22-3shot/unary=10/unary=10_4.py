
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
 
    def forward(self, x1):
        c1 = self.linear(x1)
        c2 = c1 + 3
        c3 = torch.clamp_min(c2, 0)
        c4 = torch.clamp_max(c3, 6)
        c5 = c4 / 6
        return c5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
