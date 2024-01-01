
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ReLU()
        self.pool = torch.nn.AvgPool2d(2, 2)
        self.conv1 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv5 = torch.nn.ReLU()
    def forward(self, x1, x2):
        v1 = self.pool(self.conv(x1))
        v2 = self.conv2(self.conv1(v1))
        v3 = self.conv3(torch.cat((v2, x2), 1)) # concat with x2:1
        v4 = self.conv4(v3)
        v5 = self.conv5(v4)
        return torch.abs(v5)
# Inputs to the model
x1 = torch.randn(2, 2, 32, 32)
x2 = torch.randn(2, 3, 32, 32)
