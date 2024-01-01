
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(6, 6, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(6, 6, 1, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(6, 6, 1, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(6, 6, 1, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(6, 6, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        v5 = self.conv5(v4)
        v6 = self.conv6(v5)
        v7 = v6 - 128
        return v7
# Inputs to the model
x = torch.randn(1, 3, 224, 224)
