
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=2, padding=1)
        self.conv_bn = torch.nn.BatchNorm2d(128)
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = self.conv(x2)
        v3 = v1.add_(v2)
        v4 = torch.mean(v3)
        v5 = v4.relu_()
        v6 = v5.tanh_()
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
