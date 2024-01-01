
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(10, 7, kernel_size=(3, 15), stride=(5, 5), padding=(0, 10), dilation=(2, 2), groups=1, bias=True)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(4, 10, 1, 331)
