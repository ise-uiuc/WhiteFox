
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_tran_1 = torch.nn.ConvTranspose2d(1, 1, 4, stride=4, padding=1, dilation=1, output_padding=0)
    def forward(self, x1):
        v1 = self.conv_tran_1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 10, 10)
