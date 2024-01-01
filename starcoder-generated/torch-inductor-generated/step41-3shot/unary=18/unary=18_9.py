
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 10, 5, stride=1, padding=1)
        self.batch1 = torch.nn.BatchNorm2d(10)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v2.contiguous()
        v4 = self.batch1(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 8, 256, 256)
