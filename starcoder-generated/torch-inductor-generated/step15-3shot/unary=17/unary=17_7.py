
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 5, stride=2, padding=2)
        self.conv2 = torch.nn.Conv2d(3, 3, (1, 11), stride=1, padding=(0, 5))
        self.conv3 = torch.nn.Conv2d(3, 3, kernel_size=7, groups=3)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
