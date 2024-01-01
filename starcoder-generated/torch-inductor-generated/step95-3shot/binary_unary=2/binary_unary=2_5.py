
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 96, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(96, 96, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(96, 32, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = v3 - 128
        v5 = F.relu(v1 - v4)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
