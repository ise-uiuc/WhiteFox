
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, kernel_size=1, padding=4, stride=1, bias=True)
    def forward(self, x1):
        v1 = self.conv(x1)
        v3 = v1 * v1
        v6 = v3 * v3
        v7 = v6 * 35
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 480, 640)
