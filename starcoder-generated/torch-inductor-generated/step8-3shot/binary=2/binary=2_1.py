
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.avg_pool = torch.nn.AvgPool2d(2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.avg_pool(v1 + 7.0)
        v3 = torch.flatten(v2, 1)
        v4 = v3.mean()
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
