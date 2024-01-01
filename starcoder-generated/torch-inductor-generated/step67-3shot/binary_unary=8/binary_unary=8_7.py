
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 27, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(27, 27, 1, stride=2, padding=0)
        self.conv3 = torch.nn.Conv2d(27, 27, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(27, 19, 1, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v2 + x1
        v4 = self.conv3(v3)
        v5 = torch.max(v4)
        v6 = x1 + v5
        v7 = self.conv4(v6)
        v8 = torch.relu(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 1, 96, 96)
