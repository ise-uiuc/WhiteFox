
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 1, 3, bias=False)
        self.avg_pool2d = torch.nn.AvgPool2d(3)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.avg_pool2d(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
