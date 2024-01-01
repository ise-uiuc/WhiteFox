
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x2):
        y1 = self.linear(x2)
        y2 = y1 + 3
        y3 = torch.clamp_min(y2, 0)
        y4 = torch.clamp_max(y3, 6)
        y5 = y4 / 6
        return y5

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 3)
