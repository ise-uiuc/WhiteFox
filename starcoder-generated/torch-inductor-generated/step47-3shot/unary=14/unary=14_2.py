
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(3, 3, 2, stride=2, padding=0)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(3, 3, 2, stride=2, padding=0)
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(3, 3, 2, stride=2, padding=0)
    def forward(self, x):
        v1 = self.conv_transpose_1(x)
        v2 = self.conv_transpose_2(v1)
        v3 = self.conv_transpose_3(v2)
        v4 = torch.sigmoid(v3)
        v5 = v3 * v4
        return v5
# Inputs to the model
x = torch.randn(1, 3, 4, 4)
