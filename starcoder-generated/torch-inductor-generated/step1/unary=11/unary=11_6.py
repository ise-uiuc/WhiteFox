
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(8, 64, 6, stride=2, padding=2, output_padding=1)
 
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.clamp_min(v1 + 3, 0)
        v3 = torch.clamp_max(v2, 6)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 8, 64, 20)
