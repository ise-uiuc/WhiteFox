
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(9, 61, 3, stride=1)
        self.conv1 = torch.nn.Conv2d(61, 50, 3, stride=1, padding=2, dilation=3)
        self.conv2 = torch.nn.Conv2d(50, 150, 3, stride=1, padding=2)
    def forward(self, x1):
        v1 = self.conv1_transpose(x1)
        v1 = self.conv1(v1)
        v2 = self.conv2(v1)
        v3 = v2 + 3
        v4 = torch.clamp(v3, min=0)
        v5 = torch.clamp(v4, max=6)
        v6 = v2 * v5
        v7 = v6 / 6
        return v7
# Inputs to the model
x1 = torch.randn(1, 9, 72, 72)
