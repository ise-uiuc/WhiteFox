
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, 45, stride=1, padding=22)
        self.conv2 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = torch.nn.functional.conv2d(x1, self.conv1.weight, self.conv1.bias, -18, 5, 22, 1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = torch.nn.functional.conv2d(v6, self.conv2.weight, self.conv2.bias, 10, 1, 0)
        v8 = torch.nn.functional.conv2d(v7, self.conv3.weight, self.conv3.bias, 1, 5, 1, 0)
        v9 = v8 * 0.5
        v10 = v8 * 0.7071067811865476
        v11 = torch.erf(v10)
        v12 = v11 + 1
        v13 = v9 * v12
        return v13
# Inputs to the model
x1 = torch.randn(1, 1, 112, 137)
