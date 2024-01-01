
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 8)
 
    def forward(self, x2):
        y1 = self.linear(x2)
        y2 = y1 * torch.clamp(y1, 0, 6) + 3
        y3 = y2 / 6
        return y3

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 5)
