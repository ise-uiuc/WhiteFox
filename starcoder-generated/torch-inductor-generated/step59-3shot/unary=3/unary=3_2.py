
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(5, 10, 5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(10, 20, 5, stride=2, padding=5)
        self.conv3 = torch.nn.Conv2d(20, 10, 7, stride=1, padding=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv2(v6)
        v8 = torch.mean(v7, dim=[0, 2, 3])
        v9 = self.conv3(v7)
        v10 = torch.max(v6, dim=-1).values
        return v9, v10
# Inputs to the model
x1 = torch.randn(23, 5, 48, 48)
