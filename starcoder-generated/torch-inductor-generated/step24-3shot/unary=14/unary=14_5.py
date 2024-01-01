
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_9 = torch.nn.ConvTranspose2d(64, 64, (1, 1), stride=(1, 1), padding=(2, 2), dilation=(4, 4))
        self.batch_normalization_11 = torch.nn.BatchNorm2d(64)
    def forward(self, x1):
        v1 = self.conv_transpose_9(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v3 = self.batch_normalization_11(v3)
        return v3
# Inputs to the model
x1 = torch.randn(1, 64, 16, 16)
