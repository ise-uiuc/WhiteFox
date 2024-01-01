
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(13, 9)
 
    def forward(self, x1):
        x2 = self.linear(x1)
        x3 = torch.clamp(x2 + 3.0, 0, 6)
        x4 = x3 / 6
        return x4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 13)
