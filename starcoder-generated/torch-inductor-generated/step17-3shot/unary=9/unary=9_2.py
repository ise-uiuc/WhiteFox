
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu6 = torch.nn.ReLU6()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.norm = torch.nn.BatchNorm2d(8)
    def forward(self, x1):
        v1 = self.relu6(x1)
        v2 = self.conv(v1)
        v3 = self.norm(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
