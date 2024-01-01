
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(128, 64, 3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1, bias=True, padding_mode='zeros')
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 128, 32, 272)
