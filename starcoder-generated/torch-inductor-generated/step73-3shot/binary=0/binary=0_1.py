
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 5, stride=2, padding=1)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.avg_pool(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 128, 256, 256)
