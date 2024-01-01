
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pooling = torch.nn.AvgPool2d(512, 1024)
        self.conv = torch.nn.Conv2d(3, 1000, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.pooling(x1)
        v2 = self.conv(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 2048, 1024)
