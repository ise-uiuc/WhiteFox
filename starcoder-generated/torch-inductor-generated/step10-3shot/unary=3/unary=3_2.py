
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(256, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 48, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(48, 32, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = nn.ReLU()(self.conv1(x1))
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.relu(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = nn.ReLU()(self.conv2(v6))
        v8 = v7 * 0.5
        v9 = v7 * 0.7071067811865476
        v10 = torch.relu(v9)
        v11 = v10 + 1
        v12 = v8 * v11
        v13 = nn.ReLU()(self.conv3(v12))
        v14 = v13 * 0.5
        v15 = v13 * 0.7071067811865476
        v16 = torch.relu(v15)
        v17 = v16 + 1
        v18 = v14 * v17
        v19 = nn.ReLU()(self.conv4(v18))
        return v19
# Inputs to the model
x1 = torch.randn(1, 256, 64, 64)
