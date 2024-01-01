
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1, dilation=2, groups=4)
        self.groups = 64 * 4
    def forward(self, x1):
        v1 = self.conv1(x1)
        v1 = v1.view(-1, self.groups, 1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v1
# Inputs to the model
x1 = torch.randn(1, 2592, 1, 1)
