
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
 
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 + 3
        v3 = torch.clamp(v2, min=0)
        v4 = v3 + 6
        v5 = torch.clamp(v4, max=6)
        v6 = v1 * v5
        v7 = v6 / 6
        return v7

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
