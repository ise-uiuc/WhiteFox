
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv4 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, t1, t2):
        v1 = self.conv1(x1)
        v2 = t1 + x2
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        v5 = t1 + x2
        v6 = torch.relu(v5)
        v7 = self.conv3(v6)
        v8 = v7 + v2
        v9 = torch.relu(v8)
        v10 = self.conv4(v9)
        v11 = torch.relu(t1 + v10)
        v12 = torch.cat([v11, v11, v11], axis=0)
        v13 = torch.relu(t2 + v12)
        return v13
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
t1 = torch.randn(1, 16, 64, 64)
t2 = torch.randn(1, 16, 64, 64)
