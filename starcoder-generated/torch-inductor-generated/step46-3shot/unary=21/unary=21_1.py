
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(in_channels=1, out_channels=94, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
    def forward(self, x0):
        x1 = torch.tanh(self.conv_1(x0))
        return x1
# Inputs to the model
x0 = torch.randn(1, 1, 222, 222)
