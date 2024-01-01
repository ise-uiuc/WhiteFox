
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(20, 50, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(50, 63, 2, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(63, 101, 3, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(101, 101, 7, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(101, 20, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        v5 = self.conv5(v4)
        v6 = torch.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 20, 48, 48)
