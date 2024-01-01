
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(15, 6, 3, stride=2, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(6, 6, 3, stride=2, padding=1)
        self.relu2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(6, 6, 3, stride=2, padding=1)
        self.relu3 = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.conv0(x1)
        v2 = self.relu1(v1)
        v3 = self.conv1(v2)
        v4 = self.relu2(v3)
        v5 = self.conv3(v4)
        v6 = self.relu3(v5)
        return v2
# Inputs to the model
x1 = torch.randn(1, 15, 300, 300)
