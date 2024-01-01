
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 7, 3, stride=1, padding=1, groups=1)
        self.conv2 = torch.nn.Conv2d(7, 11, 5, stride=3, padding=1, groups=2)
        self.conv3 = torch.nn.Conv2d(11, 79, 7, stride=1, padding=3, groups=1)
    def forward(self, x1, padding=None):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        if padding == None:
            padding = torch.randn()
        v4 = v3 + padding
        return v4
# Inputs to the model
x1 = torch.randn(4, 16, 64, 64)
