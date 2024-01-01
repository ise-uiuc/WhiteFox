
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(97, 41, 7, stride=3, padding=(0, 1), dilation=(3, 3))
        self.avgpool1 = torch.nn.AvgPool2d(3, stride=2, padding=(1, 1))
        self.conv2 = torch.nn.Conv2d(41, 72, 1, stride=1, padding=(2, 14))
        self.relu1 = nn.ReLU6(inplace=True)
        self.flatten = nn.Flatten()
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.avgpool1(v1)
        v3 = self.conv2(v2)
        v4 = self.relu1(v3)
        v5 = self.flatten(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 97, 79, 83)
