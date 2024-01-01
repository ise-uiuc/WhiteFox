
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_111 = torch.nn.ConvTranspose3d(7, 1, 4, stride=1, padding=0, bias=False)
    def forward(self, x1):
        v1 = self.conv_transpose_111(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 7, 8, 8, 8)
