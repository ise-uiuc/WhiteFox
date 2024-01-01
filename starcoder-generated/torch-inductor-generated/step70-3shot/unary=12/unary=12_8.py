
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 5, 3, stride=1, padding=1)
        self.conv1 = torch.nn.Conv2d(5, 3, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv1(v3)
        v5 = v3 * v4
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
