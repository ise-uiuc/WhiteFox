
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, dilation=2)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.ops.quantized.linear(v1, ((3, 1), (1, 1)), 1, (0,), False)[:, 0]
        return v2
# Inputs to the model
x = torch.randn(1, 512, 64, 64)
