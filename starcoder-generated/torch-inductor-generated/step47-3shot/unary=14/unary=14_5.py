
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(1024, 1024, 4, stride=2, padding=1)
    def forward(self, x2):
        v1 = self.conv_transpose_2(x2)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x2 = torch.randn(1, 1024, 14, 14)
