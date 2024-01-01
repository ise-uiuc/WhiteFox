
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose3d(6, 3, (7, 7, 3), stride=(3, 3, 3), padding=(2, 2, 2))
        self.conv = torch.nn.Conv3d(4, 8, kernel_size=(3, 3, 2), stride=(2, 2, 1), padding=(1, 1, 1))
        self.pool = torch.nn.MaxPool3d(5, stride=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.tanh(v1)
        v3 = self.conv(v2)
        v4 = torch.tanh(v3)
        v5 = v4.contiguous()
        v6 = self.pool(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 6, 64, 32, 64)
