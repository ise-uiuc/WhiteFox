
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(2048, 1536, 3, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose_3(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 2048, 160, 160)
