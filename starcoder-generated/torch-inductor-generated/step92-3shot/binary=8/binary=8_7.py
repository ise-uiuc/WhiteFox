
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, padding=0)
        self.conv2 = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 8, 1, padding=0)
        self.conv4 = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.conv5 = torch.nn.Conv2d(3, 8, 1, padding=0)
        self.conv6 = torch.nn.Conv2d(3, 8, 3, padding=1)
    def forward(self, x1, x2, x3, x4):
        v1 = self.conv1(x2)
        v3 = self.conv3(x4)
        v2 = self.conv2(x1)
        v4 = self.conv4(x3)
        v6 = self.conv6(x4)
        v5 = self.conv5(x3)
        v8 = v2 + v4
        v7 = v3 + v5
        return v7
# Inputs to the model
x1 = torch.randn(4, 3, 8, 8)
x2 = torch.randn(4, 3, 8, 8)
x3 = torch.randn(4, 3, 8, 8)
x4 = torch.randn(4, 3, 8, 8)
