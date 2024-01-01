
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose2 = torch.nn.ConvTranspose3d(11, 11, 2, stride=2, padding=0, dilation=1, output_padding=1)
    def forward(self, x1):
        t1 = self.conv_transpose2(x1)
        t2 = torch.sigmoid(t1)
        t3 = t1 * t2
        return t3
# Inputs to the model
x1 = torch.randn(1, 11, 78, 64, 75)
