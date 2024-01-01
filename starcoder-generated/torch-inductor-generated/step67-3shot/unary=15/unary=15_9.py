
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool1 = torch.nn.AvgPool2d(4, stride=4, padding=2)
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.pool2 = torch.nn.AvgPool2d(7, stride=7, padding=0)
        self.conv2 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(8, 6, 5, stride=1, padding=2)
    def forward(self, x1):
        v1 = self.pool1(x1)
        v2 = self.conv1(v1)
        v3 = torch.relu(v2)
        v4 = self.pool2(v3)
        v5 = self.conv2(v4)
        v6 = torch.relu(v5)
        v7 = self.conv3(v6)
        v8 = torch.relu(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
