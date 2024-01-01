
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(8, 16, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v2)
        v4 = torch.sigmoid(v3)
        v5 = v3 * v4
        return v1 * v5
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
