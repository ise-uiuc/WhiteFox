
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv2(x1)
        v3 = torch.sigmoid(v1)
        v4 = torch.sigmoid(v2)
        v5 = v1 + v2
        v6 = v3 + v4
        return v5, v6
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
