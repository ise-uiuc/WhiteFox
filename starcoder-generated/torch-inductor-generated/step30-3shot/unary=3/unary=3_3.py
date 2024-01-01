
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 1, 3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(4, 4, 9, stride=3, padding=0)
        self.conv3 = torch.nn.Conv2d(4, 2, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v6 = self.conv2(v6)
        v7 = v6 *  0.5
        v8 = v6 *  0.7071067811865476
        v9 = torch.erf(v8)
        v10 = v9 + 1
        v11 = v7 * v10
        v12 = v11 + 1
        v13 = self.conv3(v12)
        return v13
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
