
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 128)
 
    def forward(self, x1):
        x2 = self.linear(x1)
        x3 = x2 * torch.clamp(x2 + 3, min=0, max=6)
        x4 = x3 / 6
        return x4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn((10, 64))
