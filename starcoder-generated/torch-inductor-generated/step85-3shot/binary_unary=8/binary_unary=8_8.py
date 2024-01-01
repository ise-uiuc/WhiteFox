
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(1, 16, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(1, 16, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(1, 16, 1, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(1, 16, 1, stride=1, padding=0)
        self.conv6 = torch.nn.Conv2d(1, 16, 1, stride=1, padding=0)
        self.conv7 = torch.nn.Conv2d(1, 16, 1, stride=1, padding=0)
        self.conv8 = torch.nn.Conv2d(1, 16, 1, stride=1, padding=0)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(x)
        v3 = self.conv3(x)
        v4 = self.conv4(x)
        v5 = self.conv5(x)
        v6 = self.conv6(x)
        v7 = self.conv7(x)
        v8 = self.conv8(x)
        v9 = v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8
        v10 = torch.relu(v9)
        return v10
# Inputs to the model
x = torch.randn(1, 1, 64, 64)
