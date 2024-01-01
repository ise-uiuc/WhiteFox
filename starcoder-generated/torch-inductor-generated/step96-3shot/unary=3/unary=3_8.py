
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(17, 78, 6, stride=1, padding=0)
        self.conv2 = torch.nn.Conv1d(78, 91, 8, stride=1, padding=7)
        self.conv3 = torch.nn.Conv2d(91, 60, 7, stride=7, padding=0)
    def forward(self, x1):
        v1 = torch.nn.functional.elu(x1)
        v2 = self.conv(v1)
        v3 = v2 * 0.5
        v4 = v2 * 0.7071067811865476
        v5 = torch.erf(v4)
        v6 = v5 + 1
        v7 = v3 * v6
        v8 = torch.tan(v7)
        v9 = self.conv2(v8)
        v10 = v9 * 0.5
        v11 = v9 * 0.7071067811865476
        v12 = torch.erf(v11)
        v13 = v12 + 1
        v14 = v10 * v13
        v15 = self.conv3(v14)
        return v15
# Inputs to the model
x1 = torch.randn(1, 3, 86)
