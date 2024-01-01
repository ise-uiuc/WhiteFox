
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=100)
        self.conv2 = torch.nn.Conv2d(3, 8, 2, stride=1, padding=50)
        self.conv3 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=25)
        self.conv4 = torch.nn.Conv2d(3, 8, 4, stride=1, padding=13)
        self.conv5 = torch.nn.Conv2d(3, 8, 5, stride=1, padding=7)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(x)
        v3 = v1 + v2
        v4 = self.conv3(x)
        v5 = v3 + v4
        v6 = self.conv4(x)
        v7 = v5 + v6
        v8 = self.conv5(x)
        v9 = v7 + v8
        return v9
# Inputs to the model
x = torch.randn(1, 3, 32, 32)
# x = torch.randn(1, 3, 32, 32)
