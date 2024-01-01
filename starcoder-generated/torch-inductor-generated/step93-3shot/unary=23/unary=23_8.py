
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(128, 64, 7, bias=True, dilation=2)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.elu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 128, 300, 150)
