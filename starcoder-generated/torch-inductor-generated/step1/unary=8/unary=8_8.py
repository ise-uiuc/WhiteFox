
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(8, 3, 3, stride=1, padding=1)
 
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 + 3.0
        v3 = torch.nn.functional.clamp(v2, min=0.0, max=6.0)
        v4 = v3 * 6.0
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 8, 64, 64)
