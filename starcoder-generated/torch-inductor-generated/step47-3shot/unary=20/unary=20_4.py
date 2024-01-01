
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transp = torch.nn.ConvTranspose1d(9, 8, 3, stride=1, padding=0, groups=3, bias=False, dilation=2)
    def forward(self, x1):
        v1 = self.conv_transp(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 9, 16)
