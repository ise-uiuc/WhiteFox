
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100, 100)
 
    def forward(self, x1):
        x2 = self.linear(x2)
        x3 = x2 + 0.5
        return x3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(128, 100)
