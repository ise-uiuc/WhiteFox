
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(5, 5, 3, stride=2, padding=1, output_padding=1)
        self.pool1 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = self.pool1(x1)
        t1 = v1 + v2
        v3 = torch.clamp(t1, min=0)
        v4 = torch.clamp(v3, max=6)
        v5 = v1 * v4
        v6 = v5 / 6
        return v6
# Inputs to the model
x1 = torch.randn(1, 5, 28, 28)
