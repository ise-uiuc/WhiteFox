
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 16, 3, padding=1)
        self.gap = torch.nn.AvgPool2d(kernel_size=3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.nn.functional.sigmoid(v1)
        v3 = self.gap(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 6, 28, 28)
