
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(1, 4, 5, stride=2, padding=5)
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(4, 3, 5, stride=2)
    def forward(self, x1):
        v1 = self.conv_transpose_2(x1)
        v2 = torch.sigmoid(v1)
        v3 = torch.squeeze(v1, 0)
        v4 = self.conv_transpose_1(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 16, 16)
