
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(8, 32, 3, stride=1, padding=1)
 
    def forward(self, x, negative_slope):
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = v1 * (1 - v2)
        v4 = v3 * 0.9999997615814209
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 8, 64, 64)
negative_slope = 0.01
