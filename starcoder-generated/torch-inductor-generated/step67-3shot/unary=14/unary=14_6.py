
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_7 = torch.nn.ConvTranspose2d(128, 37, 4, stride=4, padding=4, dilation=3)
        self.conv_transpose_8 = torch.nn.ConvTranspose2d(37, 97, 4, stride=2, padding=0, dilation=2)
        self.conv_transpose_9 = torch.nn.ConvTranspose2d(97, 22, 4, stride=3, padding=1, dilation=4)
        self.conv_transpose_11 = torch.nn.ConvTranspose2d(22, 97, 2, stride=1, padding=0, dilation=1)
        self.conv_transpose_12 = torch.nn.ConvTranspose2d(97, 104, 1, stride=1, padding=0, dilation=1)
    def forward(self, x1):
        v1 = self.conv_transpose_7(x1)
        v1_1 = torch.sigmoid(v1)
        v1_2 = v1 * v1_1
        v2 = self.conv_transpose_8(v1_2)
        v2_1 = torch.sigmoid(v2)
        v2_2 = v2 * v2_1
        v3 = self.conv_transpose_9(v2_2)
        v3_1 = torch.sigmoid(v3)
        v3_2 = v3 * v3_1
        v4 = self.conv_transpose_11(v3_2)
        v4_1 = torch.sigmoid(v4)
        v4_2 = v4 * v4_1
        v5 = self.conv_transpose_12(v4_2)
        v5_1 = torch.sigmoid(v5)
        v5_2 = v5 * v5_1
        return v5_2
# Inputs to the model
x1 = torch.randn(1, 128, 31, 31)
