
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1= torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.conv4 = torch.nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.relu2 = torch.nn.ReLU()
        self.conv5 = torch.nn.Conv2d(64, 1, 3, stride=2, padding=1)
        self.conv6 = torch.nn.Conv2d(1,1, 1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.relu1(v3)
        v5 = self.conv4(v4)
        v6 = self.relu2(v5)
        v7 = self.conv5(v6)
        v8 = self.conv6(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
