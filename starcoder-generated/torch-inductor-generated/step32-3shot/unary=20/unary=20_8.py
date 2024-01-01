
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(2, 1, kernel_size=(2, 5), padding=(0, 2), stride=(1, 1), dilation=(1, 1), groups=1, bias=True)
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 7, 7)
