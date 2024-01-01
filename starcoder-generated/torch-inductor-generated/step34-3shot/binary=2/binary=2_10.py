
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=6, stride=3, padding=2)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = v1 - 12.8
        v3 = self.conv2(v2)
        v4 = v3 - 32.0
        v5 = self.conv3(v4)
        v6 = v5 - 14.4
        return v6
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
