
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 6)
 
    def forward(self, x):
        y = self.linear(x)
        y = y + 3
        y = torch.clamp(y, min=0, max=6)
        y = y / 6
        return y

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(10, 3)
