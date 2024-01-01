
class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(2, 4, 1, stride=1, padding=0, dilation=1)
    def forward(self, x2):
        v1 = self.conv(x2)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x2 = torch.randn(1, 2, 128, 128)
