
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(256, 256*64)
 
    def forward(self, x0):
        y0 = self.linear(x0)
        y1 = y0 + 3
        y2 = torch.clamp_min(y1, 0)
        y3 = torch.clamp_max(y2, 6)
        y4 = y3 / 6
        return y4

# Initializing the model
m = Model()

# Inputs to the model
x0 = torch.randn(100, 256)
