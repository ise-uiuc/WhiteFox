
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, 1, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(64, 128, 1, stride=2, padding=0)
        self.conv3 = torch.nn.Conv2d(128, 256, 1, stride=2, padding=0)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        return v3
# Inputs to the model
x = torch.randn(1, 1, 64, 64)
