
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 128)
 
    def forward(self, x1):
        x1 = self.linear(x1)
        x2 = x1 * torch.clamp(x1 + 3, 0, 6)
        x3 = x2 / 6
        return x3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(8, 3, 64, 64)
