
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 2, kernel_size=(19, 15), stride=1, padding=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.bn(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 32, 144, 288)
