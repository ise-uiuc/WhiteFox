
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 3, (3, 1), stride=(2, 3), padding=(0, 2), output_padding=(2, 1), dilation=(2, 3), groups=3, bias=False)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 403, 706)
