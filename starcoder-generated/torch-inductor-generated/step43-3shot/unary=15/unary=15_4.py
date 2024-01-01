
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 8, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(8, 3, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = torch.clamp(v3, min=0, max=1)
        return v4
# Inputs to the model
x1 = torch.randn(1, 4, 256, 256)
