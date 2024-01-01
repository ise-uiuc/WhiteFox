
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=7, out_channels=32, kernel_size=(1, 1))
        self.bn = torch.nn.BatchNorm2d(32, eps=0.0010000000474974513, momentum=0.8)
    def forward(self, x1):
        y1 = self.conv(x1)
        t1 = self.bn(y1)
        u1 = torch.tanh(t1)
        return u1
# Inputs to the model
x1 = torch.randn(1, 7, 14, 56)
