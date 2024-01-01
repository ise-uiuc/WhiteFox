
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 8, 3, stride=1, padding=1)
 
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 + 3
        v3 = torch.clamp(torch.clamp(v1, min=0), max=6)
        v4 = v3 / 6
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
