
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_4 = torch.nn.ConvTranspose2d(12, 523, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_transpose_5 = torch.nn.ConvTranspose2d(523, 228, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_transpose_6 = torch.nn.ConvTranspose2d(228, 72, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    def forward(self, x1):
        v1 = self.conv_transpose_4(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose_5(v3)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        v7 = self.conv_transpose_6(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 12, 8, 8)
