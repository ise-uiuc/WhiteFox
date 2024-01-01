
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 8, 5, stride=1, padding=2)
        self.conv4 = torch.nn.Conv2d(3, 8, 7, stride=1, padding=3)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(x)
        v3 = self.conv3(x)
        v4 = self.conv4(x)
        v5 = v1 + v2
        v6 = v2 + v3
        return v4 + v5 + v6
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
