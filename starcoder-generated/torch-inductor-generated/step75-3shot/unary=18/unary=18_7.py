
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), (2, 1), (1, 1), bias=False)
        self.conv2 = torch.nn.Conv2d(32, 128, (1, 7), (1, 1), (0, 0), bias=False)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v2)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 28, 100)
