
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 6)
 
    def forward(self, x1):
        h1 = self.linear(x1)
        h2 = h1 * torch.clamp(h1 + 3, 0, 6)
        h3 = h2 / 6
        return h3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(24, 32)
