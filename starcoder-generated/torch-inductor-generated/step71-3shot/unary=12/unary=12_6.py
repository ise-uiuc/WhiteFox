
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 5, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 64, 7, stride=3, padding=23)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1.sigmoid()
        v3 = v1 * v2
        v4 = self.conv2(v3)
        v5 = v4.sigmoid()
        v6 = v4 * v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
