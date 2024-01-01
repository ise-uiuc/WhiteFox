
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.conv5 = torch.nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.conv6 = torch.nn.Conv2d(3, 17, 3, stride=2, padding=1)
        self.conv7 = torch.nn.Conv2d(3, 18, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = self.conv3(x1)
        v4 = self.conv4(x1)
        v5 = self.conv5(x1)
        v6 = self.conv6(x1)
        v7 = self.conv7(x1)
        v8 = v1 + v2 + v3 + v4 + v5 + v6 + v7
        v9 = torch.relu(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
