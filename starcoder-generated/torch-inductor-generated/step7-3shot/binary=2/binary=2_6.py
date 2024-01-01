
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 4, kernel_size=1, stride=1, padding=1, bias=False)
        self.conv_1 = torch.nn.Conv2d(4, 9, kernel_size=1, stride=1, padding=1, bias=False)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv_1(v1)
        v3 = v2 - 9.2
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 64, 62)
