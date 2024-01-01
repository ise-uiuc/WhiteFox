
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_3x1 = torch.nn.ConvTranspose2d(1, 2, (3, 1), stride=(1, 1), padding=(0, 0))
        self.conv_transpose_1x3 = torch.nn.ConvTranspose2d(2, 1, (1, 3), stride=(1, 1), padding=(0, 0))
        self.sigmoid_1 = torch.nn.Identity()
        self.sigmoid_2 = torch.nn.Identity()
    def forward(self, x1):
        v0 = self.conv_transpose_3x1(x1)
        v1 = self.conv_transpose_1x3(v0)
        v2 = self.sigmoid_1(v1)
        v3 = self.sigmoid_2(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 256, 128)
