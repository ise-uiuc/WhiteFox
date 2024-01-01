
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 2, stride=2, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = v1 * 0.1
        v4 = torch.where(v2, v1, v3)
        v5 = torch.softmax(v4, 1)
        v6 = torch.where(v2, v5, v4)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 10, 10)
