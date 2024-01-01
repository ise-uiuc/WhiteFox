
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 576, 1, stride=1, padding=1)
    def forward(self, x1):
        x2 = self.conv(x1)
        x3 = torch.nn.functional.adaptive_avg_pool2d(x2, output_size=(1, 1))
        x4 = torch.cat([x3, x2], dim=1)
        x5 = torch.nn.functional.conv_transpose2d(x4, weight=None, stride=1, padding=1, output_padding=0, groups=1, dilation=1)
        x6 = torch.nn.functional.tanh(x5)
        return x6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
