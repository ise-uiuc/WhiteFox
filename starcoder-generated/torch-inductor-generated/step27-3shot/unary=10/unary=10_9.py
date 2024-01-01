
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 12)
 
    def forward(self, x1):
        x1 = self.linear(x1)
        x2 = x1 + 3
        x3 = torch.clamp(x2, -1.0, 1.0)
        x6 = x3 * 0.5
        return x6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 3)
