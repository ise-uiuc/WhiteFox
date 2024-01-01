
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size = 5, stride = 5, padding = 0, groups = 1, dilation = 2, bias = False, padding_mode = 'zeros')
    def forward(self, x1):
        v1 = self.conv1(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
