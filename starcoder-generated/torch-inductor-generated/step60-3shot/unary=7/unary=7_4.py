
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)
 
    def forward(self, x1):
        x1 = self.linear(x1)
        x2 = (torch.clamp(x1, min=0, max=6.0) + 3) / 6.0
        return x2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
