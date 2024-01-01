
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 5, stride=2, padding=0, dilation=2)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, stride=1, padding=0,  dilation=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v1 + v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
