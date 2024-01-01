
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.depthwise_conv1 = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=0, dilation=1, groups=1, bias=True)
    def forward(self, x1):
        v1 = self.depthwise_conv1(x1)
        v2 = self.depthwise_conv1(x1)
        v3 = v1 + v2
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
