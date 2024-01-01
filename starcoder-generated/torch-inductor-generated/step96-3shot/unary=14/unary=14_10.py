
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_0 = torch.nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv_transpose_4 = torch.nn.ConvTranspose2d(32, 1, kernel_size=(1, 1), stride=(2, 2), padding=(0, 0))
    def forward(self, x1):
        v1 = self.conv_transpose_0(x1)
        v2 = self.conv_transpose_1(v1)
        v3 = self.conv_transpose_2(v2)
        v4 = self.conv_transpose_3(v3)
        v5 = torch.sigmoid(v4)
        v6 = self.conv_transpose_4(v5)
        return v6
# Inputs to the model
x1 = torch.randn(4, 512, 2, 2)
