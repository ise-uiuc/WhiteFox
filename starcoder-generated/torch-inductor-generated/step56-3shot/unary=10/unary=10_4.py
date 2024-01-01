
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(256, 256)
 
    def forward(self, x1):
        y1 = self.linear(x1)
        y3 = y1 + 3
        y4 = torch.clamp_min(y3, 0)
        y5 = torch.clamp_max(y4, 6)
        y6 = y5 / 6
        return y6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2304)
