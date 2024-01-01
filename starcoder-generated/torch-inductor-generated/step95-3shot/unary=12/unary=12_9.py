
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 4, stride=4, padding=4)
        self.conv2 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 3, 2, stride=2, padding=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = F.relu(v3)
        v5 = F.max_pool2d(v4, 4, 4, 1, 1)
        v6 = self.conv2(v5)
        v7 = torch.sigmoid(v6)
        v8 = v6 * v7
        v9 = F.leaky_relu(v8)
        v10 = F.avg_pool2d(v9, 1, 1, 2, 2)
        v11 = self.conv3(v10)
        v12 = torch.sigmoid(v11)
        v13 = v11 * v12
        return v13
# Inputs to the model
x1 = torch.randn(1, 3, 160, 160)
