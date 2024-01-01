
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1000)
 
    def forward(self, x1):
        y1 = self.linear(x1)
        y2 = y1 * torch.clamp(min=0, max=6, y1 + 3)
        y3 = y2 / 6
        return y3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
