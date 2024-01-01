
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 256, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(256, 64, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(64, 256, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        return v4
# Inputs to the model
x1 = torch.randn(4, 3, 320, 320)
