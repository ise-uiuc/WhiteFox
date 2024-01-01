
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(1, 8, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(1, 8, 3, stride=1, padding=2)
        self.conv4 = torch.nn.Conv2d(1, 8, 3, stride=1, padding=3)
        self.conv5 = torch.nn.Conv2d(1, 8, 3, stride=2, padding=4)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = self.conv3(x1)
        v4 = self.conv4(x1)
        v5 = self.conv5(x1)
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
