
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 32, 7, stride=2)
        self.conv3 = torch.nn.Conv2d(32, 16, 3, stride=2)
        self.conv4 = torch.nn.Conv2d(16, 32, 3, stride=2)
        self.conv5 = torch.nn.Conv2d(32, 32, 7, stride=1, padding=3)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        v5 = self.conv5(v4)
        return v5
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
