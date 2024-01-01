
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
 
    def forward(self, x):
        x1 = self.linear(x)
        x2 = torch.clamp(x1 + 3, min=0, max=6)
        x3 = x2 / 6
        return x3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 10)
