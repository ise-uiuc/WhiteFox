
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(19, 9, 5, stride=1, padding=2)
        self.avg_pool2d = torch.nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 19, 5, stride=2, padding=2)
        self.gelu = torch.nn.GELU()
    def forward(self, x1):
        v1 = self.conv2d(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.avg_pool2d(v6)
        v8 = self.conv_transpose(v7)
        v9 = self.gelu(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 19, 64, 64)
