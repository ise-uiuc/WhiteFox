
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 1, 1, stride=2)
        self.conv2 = torch.nn.Conv2d(4, 1, 1, stride=2)
        self.conv3 = torch.nn.Conv2d(4, 1, 1, stride=2)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = 1 + v1
        v5 = torch.randn(int(v2.size(3)), int(v2.size(2)))
        v3 = torch.randn(10)
        v4 = v3 + v2
        v11 = v5 * v4
        v10 = self.conv2(x2)
        v6 = 3.2 + v5
        v7 = v6 * v11
        v8 = v7 + v10
        return v8
# Inputs to the model
x1 = torch.randn(1, 4, 16, 16)
x2 = torch.randn(1, 4, 16, 16)
