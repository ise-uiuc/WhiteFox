
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.sigmoid1 = torch.nn.Sigmoid()
        self.conv2 = torch.nn.Conv2d(8, 9, 1, stride=1, padding=1)
        self.sigmoid2 = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.sigmoid1(v1)
        v3 = self.conv2(v2)
        v4 = self.sigmoid2(v3)
        v5 = v2 * v4
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
