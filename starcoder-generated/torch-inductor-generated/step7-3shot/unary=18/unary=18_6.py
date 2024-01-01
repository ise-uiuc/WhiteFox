
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv = torch.nn.Conv2d(3, 64, 1, stride=1, padding=1, dilation=2)
    def forward(self, x1):
        v1 = self.pool(x1)
        v2 = self.conv(v1 + 1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 100, 100)
