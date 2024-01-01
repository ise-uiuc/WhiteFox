
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_6 = torch.nn.ConvTranspose2d(15, 34, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.conv_transpose_7 = torch.nn.ConvTranspose2d(34, 39, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    def forward(self, x1):
        v1 = self.conv_transpose_6(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose_7(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 15, 7, 7)
