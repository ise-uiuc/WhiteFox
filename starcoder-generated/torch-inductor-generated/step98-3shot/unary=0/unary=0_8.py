
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(100, 64, 15, stride=1, padding=0)
        self.conv1 = torch.nn.Conv2d(64, 256, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv0(x1)
        v2 = self.conv1(v1)
        return v2
# Inputs to the model
x1 = torch.randn(100, 100, 32, 32)
