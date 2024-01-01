
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = lambda x: torch.nn.Conv2d(3, 8, x, stride=1, padding=1)
 
    def forward(self, x):
        v1 = self.conv(3)(x)
        v2 = self.conv(1)(v1)
        v3 = self.conv(3)(v1)
        v4 = torch.cat((v1, v3, v2), dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
