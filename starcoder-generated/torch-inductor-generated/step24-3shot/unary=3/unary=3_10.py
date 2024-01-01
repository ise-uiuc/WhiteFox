
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 5, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(5, 5, 1, stride=1, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v11 = self.conv1(x2)
        v12 = v11 * 0.5
        v13 = v11 * 0.7071067811865476
        v14 = torch.erf(v13)
        v15 = v14 + 1
        v16 = v12 * v15
        v17 = torch.sum(v6 + v16)
        return v17
# Inputs to the model
x1 = torch.randn(1, 2, 1, 15)
x2 = torch.randn(1, 2, 15, 1)
