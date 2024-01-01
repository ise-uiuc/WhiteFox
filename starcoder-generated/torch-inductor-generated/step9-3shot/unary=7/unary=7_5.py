
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(20, 256)
 
    def forward(self, x1):
        z1 = self.fc(x1)
        z2 = z1 * torch.clamp(min=0, max=6, z1 + 3)
        z3 = z2/6
        return z3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(100, 20)
