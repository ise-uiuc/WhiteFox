
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
 
    def forward(self, x):
        f = torch.nn.functional.leaky_relu
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = v1
        v4 = v2 * v3
        return f(__output__, negative_slope = 1e-2)

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 128, 32, 32)
