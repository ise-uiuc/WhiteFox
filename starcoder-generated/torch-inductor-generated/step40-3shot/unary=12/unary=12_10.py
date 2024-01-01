
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.avg_pool2d = torch.nn.AvgPool2d(kernel_size=1, stride=1, padding=0, ceil_mode=True, count_include_pad=True)
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.avg_pool2d(x1)
        v2 = self.conv(v1)
        v3 = self.sigmoid(v2)
        v4 = torch.mul(x1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 32, 16)
