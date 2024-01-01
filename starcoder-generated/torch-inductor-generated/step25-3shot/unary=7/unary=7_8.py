
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8, 1, stride=1, padding=1)
 
    def forward(self, x1):
        y1 = self.linear(x1)
        y2 = y1 * torch.clamp(y1 + 3, 0, 6)
        y3 = y2 / 6
        return y3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
