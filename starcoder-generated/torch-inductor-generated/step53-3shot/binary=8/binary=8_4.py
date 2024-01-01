
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(x)
        v3 = v1 + v2
        v4 = self.conv3(v3)
        v5 = self.conv4(v3)
        v6 = v4 + v5
        return v6
# Inputs to the model
x = torch.randn(1, 3, 32, 32)
