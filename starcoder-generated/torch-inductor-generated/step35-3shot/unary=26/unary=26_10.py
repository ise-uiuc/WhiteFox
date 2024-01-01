
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.negative_slope = 0.5
        self.conv_t = torch.nn.ConvTranspose2d(1, 7, 7, stride=2, padding=0, output_padding=1, bias=True)
        self.conv = torch.nn.Conv2d(7, 21, 1, stride=1, bias=True)
    def forward(self, x):
        v1 = self.conv_t(x)
        v2 = v1 * self.negative_slope
        v3 = torch.relu(v2)
        v4 = self.conv(v3)
        v5 = v4 > 0
        v6 = v4 * -1.42
        v7 = torch.where(v5, v4, v6)
        return v7
# Inputs to the model
x = torch.randn(4, 1, 224, 224, device='cuda')
