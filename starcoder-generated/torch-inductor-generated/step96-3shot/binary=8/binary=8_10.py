
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 3, kernel_size=3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(9, 3, kernel_size=3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(10, 3, kernel_size=3, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(10, 3, kernel_size=3, stride=2, padding=1)
        self.conv5 = torch.nn.Conv2d(11, 3, kernel_size=3, stride=2, padding=1)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = self.conv3(x3)
        v4 = self.conv4(x3)
        v5 = self.conv5(x3)
        v6 = v1 + v2 + v3
        return (v6, v3 + v4, v5 + v3)
# Inputs to the model
x1 = torch.randn(1, 8, 20, 20)
x2 = torch.randn(1, 9, 20, 20)
x3 = torch.randn(1, 10, 20, 20)
