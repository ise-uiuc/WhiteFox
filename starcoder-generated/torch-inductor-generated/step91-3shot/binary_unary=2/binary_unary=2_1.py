
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 5, stride=2, padding=2, dilation=2, groups=1)
        self.convt = torch.nn.ConvTranspose2d(16, 16, 5, stride=2, padding=2, output_padding=1, groups=1, bias=True)
        self.relu = torch.nn.ReLU(inplace=False)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.convt(v1)
        v3 = self.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 16, 28, 28)
