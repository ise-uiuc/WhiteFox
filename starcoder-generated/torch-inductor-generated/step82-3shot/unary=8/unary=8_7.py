
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose3d(32, 3, 6, stride=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.transpose(v1, 1, 0)
        return v2
# Inputs to the model
x1 = torch.randn(32, 7, 26, 16)
