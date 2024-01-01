
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 2, 5)
        self.conv2 = torch.nn.Conv2d(2, 3, 5)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v1 * v2
        v5 = v1 + v2
        v7 = v1 - v2
        v9 = v3 + v5
        v11 = v3 - v5
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
