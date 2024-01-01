
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 3, 5, stride=4, padding=0)
        self.conv2 = torch.nn.Conv2d(3, 6, 1, stride=4, padding=0)
        self.conv3 = torch.nn.Conv2d(6, 7, 3, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv2(v6)
        v8 = self.conv3(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 4, 56, 56)
