
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(
            kernel_size=2,
            stride=1,
            padding=1
        )
        self.conv = torch.nn.Conv2d(3, 9, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.pool(x1)
        v2 = self.conv(v1)
        v3 = v2 + 3
        v4 = torch.clamp(v3, min=0, max=6)
        v5 = v2 * v4
        v6 = v5 / 6
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
