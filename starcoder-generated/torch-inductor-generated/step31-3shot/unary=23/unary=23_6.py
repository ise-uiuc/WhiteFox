
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(9, 36, kernel_size=2, stride=2, padding=0, dilation=1, groups=1, bias=True)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(36, 18, kernel_size=2, stride=2, padding=0, dilation=1, groups=1, bias=True)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(18, 9, kernel_size=2, stride=2, padding=0, dilation=1, groups=1, bias=True)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v3 = self.conv_transpose2(v1)
        v5 = self.conv_transpose3(v3)
        v2 = torch.tanh(v5)
        return v2
# Inputs to the model
x1 = torch.randn(1, 9, 224, 224)
