
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1056, 3072, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=1, bias=False, dilation=1)
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1056, 33, 33)
