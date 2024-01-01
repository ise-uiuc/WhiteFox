
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(8, 1, 3, stride=1, padding=1, output_padding=0, groups=1, dilation=1, bias=True, padding_mode='zeros')
     
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = v1 * 0.01
        v4 = v2 * v3
        return torch.where(v2, v1, v4)

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 8, 64, 64)
