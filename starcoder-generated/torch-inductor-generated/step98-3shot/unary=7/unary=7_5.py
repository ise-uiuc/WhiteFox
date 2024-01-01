
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 16)
 
    def forward(self, x1):
        a1 = self.linear(x1)
        a2 = torch.clamp(a1 + 3, 0, 6)
        a3 = a2 / 6
        return a3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64, 1, 1)
