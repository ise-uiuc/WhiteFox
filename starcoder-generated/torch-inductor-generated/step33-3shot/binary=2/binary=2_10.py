
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(128, 64, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 64, 2, stride=1, padding=1)
        self.conv3 = torch.nn.Conv3d(64, 2, 3, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = v3 - 5
        return v4
# Inputs to the model
x = torch.randn(1, 128, 8, 8, 8)
