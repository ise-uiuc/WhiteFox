
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0, padding_mode="reflect")
        self.conv2 = torch.nn.Conv2d(8, 8, 1, stride=3, groups=2, dilation=2, padding=0, padding_mode="replicate")
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
