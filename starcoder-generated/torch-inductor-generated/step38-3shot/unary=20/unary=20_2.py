
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose3d(2, 8, kernel_size=(17, 3, 3), stride=(2, 2, 6), bias=True, stride=1, dilation=(3, 1, 2), padding=(1, 2, 14), output_padding=(13, 5, 0), groups=2, padding_mode='zeros', weight_norm=None)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 4, 3, 3)
