
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu6 = torch.nn.ReLU6()
        self.conv = torch.nn.Conv2d(6, 16, 7, stride=1, dilation=7)
    def forward(self, x1):
        v1 = self.relu6(x1)
        v2 = v1[:, 3:14, 7:16, :]
        v3 = self.conv(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 6, 64, 64)
