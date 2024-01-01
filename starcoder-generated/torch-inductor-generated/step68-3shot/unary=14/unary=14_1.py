
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_15 = torch.nn.ConvTranspose2d(2, 144, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose_15(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 224, 224)
