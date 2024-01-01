
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 8, 5, stride=2, padding=2, dilation=2)
        self.conv2 = torch.nn.Conv2d(8, 3, 3, stride=2, padding=2, dilation=2)
        self.conv3 = torch.nn.Conv2d(3, 2, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 0.5
        v3 = self.conv3(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 41, 41)
