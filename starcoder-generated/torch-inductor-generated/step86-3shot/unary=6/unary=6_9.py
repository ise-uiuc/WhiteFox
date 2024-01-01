
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.ConvTranspose2d(8, 8, 4, 2, 0, 1)
        self.conv3 = torch.nn.Conv2d(8, 8, 3, stride=2, padding=0)
        self.conv4 = torch.nn.Conv2d(8, 8, 3, stride=2, padding=0)
        self.conv5 = torch.nn.Conv2d(8, 8, 3, stride=2, padding=0)
        self.conv6 = torch.nn.Conv2d(8, 8, 3, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(x1)
        v4 = v3 * v2
        v5 = v4 + v3
        v6 = torch.tanh(v5)
        v7 = self.conv4(v6)
        v8 = self.conv5(v7)
        v9 = v8 * x1
        v10 = v9 + x1
        v11 = torch.tanh(v10)
        v12 = self.conv6(v11)
        v13 = v12 * x1
        v14 = v13 + x1
        return v14
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
