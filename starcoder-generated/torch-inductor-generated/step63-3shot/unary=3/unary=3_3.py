
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(8, 3, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(4, 4, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(x1)
        return v3 * v4
# Inputs to the model
x1 = torch.randn(1, 4, 8, 8)
