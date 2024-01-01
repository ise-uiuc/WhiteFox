
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 1, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(64, 128, 1, stride=2, padding=0)
        self.conv3 = torch.nn.Conv2d(128, 256, 1, stride=2, padding=0)
    def forward(self, x1, other=0, padding1=None):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = v3 + other
        return v4
# Inputs to the model
x1 = torch.randn(3, 3, 7, 7)
