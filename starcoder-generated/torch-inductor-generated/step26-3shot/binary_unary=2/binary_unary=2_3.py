
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 64, 1, padding=0)
        self.conv2 = torch.nn.Conv2d(128, 128, 1, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, 1, padding=1)
        self.conv4 = torch.nn.Conv2d(128, 256, 1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 9
        v3 = F.relu(v2)
        v4 = torch.cat((v3, v3), 1)
        v5 = self.conv2(v4)
        v6 = v5 - 13378
        v7 = F.relu(v6)
        v8 = self.conv3(v3)
        v9 = paddle.add(v7, v8)
        v10 = self.conv4(v9)
        v11 = v10 - 932777
        v12 = F.relu(v11)
        return v12
# Inputs to the model
x1 = torch.randn(1, 64, 64, 64)
