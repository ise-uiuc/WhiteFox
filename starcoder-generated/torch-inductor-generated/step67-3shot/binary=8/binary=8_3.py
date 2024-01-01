
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 3, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 3, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(8, 3, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(8, 3, 3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(8, 3, 3, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(8, 3, 3, stride=1, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = self.conv3(x1)
        v4 = self.conv4(x2)
        v5 = self.conv5(x2)
        v6 = self.conv6(x2)
        v7 = v3 + v4
        v8 = v6 + v2
        v9 = v7 + v5
        v10 = v1 + v9
        return v10
# Inputs to the model
x1 = torch.randn(1, 8, 32, 32)
x2 = torch.randn(1, 8, 32, 32)
