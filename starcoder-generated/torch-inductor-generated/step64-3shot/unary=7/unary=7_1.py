
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 4)
 
    def forward(self, x):
        y1 = self.linear(x)
        y2 = y1 * torch.clamp(y1 + 3, min=0, max=6)
        y3 = y2 / 6
        return y3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 2)
