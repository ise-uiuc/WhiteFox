
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_6 = torch.nn.ConvTranspose2d(64, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 1))
        self.conv_transpose_7 = torch.nn.ConvTranspose2d(64, 64, kernel_size=(5, 5), stride=(3, 3), padding=(3, 2))
        self.conv_transpose_9 = torch.nn.ConvTranspose2d(64, 64, kernel_size=(5, 5), stride=(2, 2), padding=(3, 1), output_padding=(1, 0))
        self.conv_transpose_10 = torch.nn.ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(3, 1))
        self.conv_transpose_11 = torch.nn.ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(3, 2), padding=(1, 3))
    def forward(self, x1):
        v1 = self.conv_transpose_9(x1)
        v2 = self.conv_transpose_7(v1)
        v4 = self.conv_transpose_10(v2)
        v5 = self.conv_transpose_11(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 128, 2, 1)
