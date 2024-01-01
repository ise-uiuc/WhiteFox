
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv5 = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=0)
        self.conv6 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv7 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
    def forward(self, x1, x2, x3, x4, x5):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = v1 + v2
        v4 = self.conv3(x3)
        v5 = self.conv4(x4)
        v6 = v3 + v5
        v7 = v6 + self.conv5(x5)
        v8 = self.conv6(v7)
        v9 = v8.squeeze(dim=1)
        v10 = self.conv7(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 63, 63)
x3 = torch.randn(1, 3, 61, 61)
x4 = torch.randn(1, 3, 60, 60)
x5 = torch.randn(1, 3, 1, 1)
