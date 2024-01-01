
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 24, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(24, 48, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(48, 64, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv2(v1)
        v3 = torch.relu(v2)
        v4 = self.conv3(v3)
        v5 = torch.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
