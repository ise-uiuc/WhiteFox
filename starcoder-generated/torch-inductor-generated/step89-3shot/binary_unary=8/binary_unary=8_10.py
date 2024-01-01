
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 128, 1, stride=2, padding=1)
        self.avgpool1 = torch.nn.AvgPool2d(1)
        self.conv2 = torch.nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.avgpool2 = torch.nn.AvgPool2d(1)
        self.conv3 = torch.nn.Conv2d(128, 1, 1, stride=1, padding=0)
        torch.nn.init.normal_(self.conv3.weight, mean=0.0, std=0.01)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.avgpool1(v1)
        v3 = self.conv2(v2)
        v4 = self.avgpool2(v3)
        v5 = self.conv3(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
