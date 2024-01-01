
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(1, 8, 4, stride=2, padding=0)
        self.conv_transpose_1_1 = torch.nn.ConvTranspose2d(8, 16, 4, stride=2, padding=0)
    def forward(self, x):
        v1 = self.conv_transpose_1(x)
        v1_1 = self.conv_transpose_1_1(v1)
        v2 = torch.sigmoid(v1_1)
        v3 = v1_1 * v2
        return v3
# Inputs to the model
x = torch.randn(2, 1, 2, 2)
