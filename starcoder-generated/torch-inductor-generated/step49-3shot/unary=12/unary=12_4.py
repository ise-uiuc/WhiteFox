
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 3, 2, 1)
        self.conv2 = torch.nn.Conv2d(6, 18, 3, 1, 1)
        self.conv3 = torch.nn.Conv2d(18, 3, 3, 1, 1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv2(v3)
        v5 = self.conv3(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
