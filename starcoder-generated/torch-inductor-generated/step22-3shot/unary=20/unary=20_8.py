
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 1, kernel_size=(3, 3), stride=(2, 2), bias=False, padding=(1, 1))
        self.conv_b = torch.nn.Conv2d(1, 1, kernel_size=1, stride=1, bias=False, padding=0, dilation=1, groups=1)
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = self.conv_b(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 64, 256)
