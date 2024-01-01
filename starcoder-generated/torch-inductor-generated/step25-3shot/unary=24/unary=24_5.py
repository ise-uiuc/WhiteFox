
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.relu1 = torch.nn.LeakyReLU(-0.01)
        self.bn1 = torch.nn.BatchNorm2d(8)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.relu1(v1)
        v3 = self.bn1(v2)
        v4 = v2 > 0
        v5 = v2 * -0.1
        v6 = torch.where(v4, v2, v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 8, 128, 128)
