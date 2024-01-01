
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(3, 2)
 
    def forward(self, x1):
        y1 = self.lin(x1)
        y2 = y1 + 3
        y3 = torch.clamp_min(y2, 0)
        y4 = torch.clamp_max(y3, 6)
        y5 = y4 / 6
        return y5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
