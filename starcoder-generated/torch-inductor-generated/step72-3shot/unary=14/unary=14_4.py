
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(3, 64, kernel_size=2, stride=(2, 2), bias=False)
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(64, 2, kernel_size=(1, 1), stride=(1, 1))
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose_3(v3)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 1048, 1048)
