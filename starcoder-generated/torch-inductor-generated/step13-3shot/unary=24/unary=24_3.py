
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(3, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = v1 * 0.1
        v4 = torch.where(v2, v1, v3)
        v5 = self.pool(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
