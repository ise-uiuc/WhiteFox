
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 1, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 2, 1, stride=1, padding=1)
    def forward(self, x2):
        v1 = self.conv1(x2)
        v2 = v1.sigmoid()
        v3 = self.conv2(v2)
        v3 = v3.sigmoid()
        v4 = v1 * v2 * v3
        return v4
# Inputs to the model
x2 = torch.randn(1, 3, 256, 64)
