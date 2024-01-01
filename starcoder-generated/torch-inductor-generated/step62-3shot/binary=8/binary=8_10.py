
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2)
        self.conv2 = nn.Conv2d(3, 64, 3, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 2, stride=2)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(x)
        v3 = self.conv3(v2)
        v4 = v1 + v3
        return v4
# Inputs to the model
x = torch.randn(1, 3, 32, 32)
