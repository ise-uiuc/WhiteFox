
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_71 = torch.nn.ConvTranspose2d(53, 53, 1, stride=1, padding=0)
        self.conv_transpose_70 = torch.nn.ConvTranspose2d(61, 75, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose_71(x1)
        v2 = self.conv_transpose_70(v1)
        v3 = torch.sigmoid(v2)
        v4 = v1 * v3
        return v4
# Inputs to the model
x1 = torch.randn(1, 61, 20, 20)
