
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 5, stride=1, padding=0, groups=3, dilation=2, bias=False)
        self.conv2 = torch.nn.Conv2d(3, 8, 5, stride=1, padding=0, groups=3, dilation=2, bias=True)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = torch.relu(v1)
        v4 = v3 + v2
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 28, 28)
