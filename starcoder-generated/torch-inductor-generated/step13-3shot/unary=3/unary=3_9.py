
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(5, 10, 4, stride=(3, 2), padding=2)
        self.conv2 = torch.nn.Conv2d(10, 5, 4)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = v6 + 0.9897788426021394
        return v7
# Inputs to the model
x1 = torch.randn(1, 5, 32, 32)
