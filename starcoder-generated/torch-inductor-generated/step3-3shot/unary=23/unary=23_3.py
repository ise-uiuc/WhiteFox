
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = torch.nn.Conv2d(1, 1, 1, bias=False, padding=(0, 0), stride=(1, 1))
        self.conv_transpose_1 = torch.nn.ConvTranspose1d(1, 1, 1, bias=True, padding=(0, 0), stride=(1, 1))
    def forward(self, x1):
        v1 = self.conv2d_1(x1)
        v2 = self.conv_transpose_1(v1)
        return v2
# Inputs to the model
x1 = torch.randn(2, 1, 10, 10)
