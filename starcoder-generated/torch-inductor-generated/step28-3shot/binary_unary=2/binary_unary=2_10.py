
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, 16, stride=2, padding=3)
        self.conv2 = torch.nn.Conv2d(8, 4, 32, stride=2, padding=3)
        self.conv3 = torch.nn.Conv2d(4, 32, 32)
    def forward(self, x1):
        v1 = self.conv1(x1)
        r1 = F.relu(v1)
        v2 = self.conv2(r1)
        r2 = F.relu(v2)
        v3 = self.conv3(r2)
        r3 = F.relu(v3)
        v4 = r3 - 0.6
        v5 = torch.sigmoid(v4)
        v6 = r3 - 2
        v7 = v6 * v5
        v8 = torch.tanh(v5)
        v9 = torch.matmul(v7, v8)
        # Above line should be rearranged to v9 = v7.mm(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 1, 128, 128)
