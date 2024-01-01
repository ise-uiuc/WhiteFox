
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 96, 1, stride=3, padding=1)
        self.conv2 = torch.nn.Conv2d(96, 128, 1, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(128, 256, 1, stride=2, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1) - 256
        v3 = self.conv3(v2)
        return (v1, v2, v3)
# Inputs to the model
x = torch.randn(1, 3, 224, 244)
