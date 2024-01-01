
class Model_2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    def forward(self, x3):
        v1 = self.conv_transpose_2(x3)
        v2 = torch.sigmoid(v1)
        return v1, v2
# Inputs to the model
x3 = torch.randn(1, 512, 64, 64)
