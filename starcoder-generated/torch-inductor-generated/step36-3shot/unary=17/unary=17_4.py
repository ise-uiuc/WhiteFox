
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(1, 16, 1, stride=2, padding=1)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(16, 1, 1, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = self.conv_transpose_2(v1)
        v3 = torch.reshape(v2, (1, 16, 8, 8))
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 8, 8)
