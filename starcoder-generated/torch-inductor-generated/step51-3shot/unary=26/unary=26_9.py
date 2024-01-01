
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.functional.conv_transpose2d(kernel_size=(2, 2), stride=1, output_padding=0, bias=False, input=None, padding=0, groups=None, D=2, H=25, W=16, in_channels=14, out_channels=4, weight=torch.Size([4, 12, 2, 2]))
    def forward(self, x10):
        w1 = self.conv_t(x10)
        y = w1 > 0
        z = w1 * 5.8057
        v4 = torch.where(y, w1, z)
        return v4
# Inputs to the model
x10 = torch.randn(10, 14, 25, 16)
