
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(4, 3, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = torch.mul(v1, v2)
        v4 = self.conv2(v3)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
