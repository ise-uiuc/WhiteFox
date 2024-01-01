
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(128, 1024, 1, stride=1, padding=0)
        self.conv1 = torch.nn.Conv2d(1024, 1024, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(1024, 1024, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(1024, 128, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv1(v1)
        v3 = self.conv2(v2)
        v4 = self.conv3(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 128, 32, 32)
