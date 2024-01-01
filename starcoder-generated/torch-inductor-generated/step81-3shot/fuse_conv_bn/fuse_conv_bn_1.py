
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(256, 256, 1)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(256, 256, 1)
        self.relu2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(256, 256, 1)
        self.relu3 = torch.nn.ReLU()
        self.conv0 = torch.nn.Conv2d(256, 32, 3, padding=1, stride=2, bias=False)
        self.bn0 = torch.nn.BatchNorm2d(32)
    def forward(self, x3):
        x3 = self.relu1(self.conv1(x3) + self.conv2(x3) + self.conv3(x3))
        x3 = self.bn0(self.conv0(x3))
        return x3
# Inputs to the model
x3 = torch.randn(1, 256, 64, 64)
