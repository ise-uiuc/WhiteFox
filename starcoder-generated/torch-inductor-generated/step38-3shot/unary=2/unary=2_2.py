
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(1, 4, 7, stride=2, padding=1)
        self.conv_transpose_1_pointwise = torch.nn.Conv2d(4, 4, 1)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(4, 8, 4, stride=(2, 2), padding=1)
        self.conv_transpose_2_pointwise = torch.nn.Conv2d(8, 8, 1)
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(8, 1, 4, stride=2)
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = self.conv_transpose_1_pointwise(v1)
        v3 = self.conv_transpose_2(v2)
        v4 = self.conv_transpose_2_pointwise(v3)
        v5 = self.conv_transpose_3(v4)
        return v5
# Inputs to the model
x1 = torch.randn(5, 1, 24, 33)
