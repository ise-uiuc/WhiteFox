
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 1, stride=2, dilation=1, padding=1)
        self.pointwise = torch.nn.Conv2d(1, 1, 1, stride=2, dilation=1, padding=1)
    def forward(self, x1):
        # The input tensor x1 is from 1 channel to 3 channels.
        v1 = self.conv(x1)
        # The input tensor x1 is from 3 channels to 1 channel.
        v2 = self.pointwise(x1)
        v3 = v2 * v1
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
