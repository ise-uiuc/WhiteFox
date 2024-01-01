
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
        self.relu6 = torch.nn.ReLU6()
    def forward(self, x1):
        v1 = self.relu6(self.conv(x1) + 3)
        v2 = v1 * 0.16666666666666666
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
