
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(6, 50, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(50, 6, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v3 = self.conv_transpose2(v1)
        v2 = torch.tanh(v3)
        return v2
# Inputs to the model
x1 = torch.randn(1, 6, 224, 224)
