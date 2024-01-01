
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = F.sigmoid(v2)
        v4 = torch.sigmoid(v2)
        v5 = v2 * v3 + v2 * v4
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
