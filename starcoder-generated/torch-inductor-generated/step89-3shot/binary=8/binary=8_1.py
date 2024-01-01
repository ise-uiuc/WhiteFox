
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(8, 32, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(8, 32, 3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(32, 128, 1, stride=1, padding=0)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(x)
        v3 = self.conv3(v1)
        v4 = self.conv4(v2)
        return self.conv5(v3 + v4)
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
