
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(16, 32, 1, stride=1)
        self.conv1 = torch.nn.Conv2d(32, 16, 1, stride=1)
        self.conv2 = torch.nn.Conv2d(32, 8, 1, stride=1)
        self.conv3 = torch.nn.Conv2d(16, 16, 1, stride=1)
    def forward(self, x2):
        v1 = self.conv0(x2)
        v2 = v1 - 0.67935
        v3 = F.relu(v2)
        v4 = self.conv1(v3)
        v5 = v4 - 0.9798
        v6 = F.relu(v5)
        v7 = self.conv2(v6)
        v8 = self.conv3(v7)
        v9 = v8 - -11.23355
        return v9
# Inputs to the model
x2 = torch.randn(1, 16, 846, 391)
