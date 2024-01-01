
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(2, 4, 3, stride=2, padding=1)
        self.conv1 = torch.nn.Conv2d(4, 7, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(7, 10, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(10, 13, 3, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(13, 25, 3, stride=2, padding=1)
    def forward(self, x1):
        v0 = self.conv0(x1)
        v1 = v0 * 0.5
        v2 = v0 * 0.7071067811865476
        v7 = self.conv1(v2)
        v17 = self.conv2(v7)
        v31 = self.conv3(v17)
        v54 = self.conv4(v31)
        return v54
# Inputs to the model
x1 = torch.randn(1, 2, 111, 111)
