
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(2, 29, 1, stride=1, padding=0)
        self.relu0 = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(29, 80, 1, stride=1, padding=0)
        self.bn1 = torch.nn.BatchNorm2d(80)  
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(80, 3, 1, stride=1, padding=0)
        self.relu2 = torch.nn.ReLU()
    def forward(self, x6):
        v1 = self.conv0(x6)
        v2 = self.relu0(v1)
        v3 = self.conv1(v2)
        v4 = self.bn1(v3)
        v5 = self.relu1(v4)
        v6 = self.conv2(v5)
        v7 = self.relu2(v6)
        return v7
# Inputs to the model
x6 = torch.randn(1, 2, 51, 17)
