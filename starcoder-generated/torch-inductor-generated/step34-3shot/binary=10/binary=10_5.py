
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(23, 42)
 
    def forward(self, x1, x2=None):
        x3 = self.linear(x1)
        x4 = x3 + x2
        return x4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(128, 23)
x2 = torch.randn(128, 42)
