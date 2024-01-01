
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(10, 5, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(5, 5, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(5, 3, 3, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(3, 10, 1, stride=1, padding=0)
        self.conv5 = torch.nn.ConvTranspose2d(10, 1, 2, stride=2, padding=0)
        self.conv6 = torch.nn.Conv2d(1, 3, 2, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv2(v6)
        v8 = self.conv3(v7)
        v9 = self.conv4(v8)
        v10 = self.conv5(v9)
        v11 = self.conv6(v10)
        return v11
# Inputs to the model
x1 = torch.randn(1, 10, 21, 21)
