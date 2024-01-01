
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 32)
 
    def forward(self, x1, x2):
        z1 = self.fc(x1)
        z2 = z1 + x2
        return z2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
x2 = torch.randn(1, 32)
__output__, 