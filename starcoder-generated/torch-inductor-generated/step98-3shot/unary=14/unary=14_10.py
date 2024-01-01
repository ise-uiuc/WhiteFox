
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(3, 3, 7, stride=7, _output_padding=0, bias=False)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(3, 3, 7, stride=1, _output_padding=0, bias=True)
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = self.conv_transpose_2(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 30, 30)
