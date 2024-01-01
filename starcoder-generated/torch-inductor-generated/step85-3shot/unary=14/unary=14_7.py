
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(16, 8, 7, stride=2, padding=1, output_padding=0)
        self.conv_transpose_4 = torch.nn.ConvTranspose2d(8, 6, 3, stride=2, padding=1, output_padding=1)
        self.conv_transpose_5 = torch.nn.ConvTranspose2d(6, 19, 16, stride=2, padding=5, output_padding=1)
        self.conv_transpose_6 = torch.nn.ConvTranspose2d(19, 26, 27, stride=1, padding=5, output_padding=1)
        self.conv_transpose_7 = torch.nn.ConvTranspose2d(26, 5, 6, stride=2, padding=2, output_padding=1)
        self.conv_transpose_8 = torch.nn.ConvTranspose2d(5, 2, 6, stride=1, padding=0, output_padding=1)
        self.relu_32 = torch.nn.ReLU(inplace=False)
    def forward(self, x1):
        v1 = self.conv_transpose_3(x1)
        v2 = torch.nn.functional.relu(v1)
        v3 = self.conv_transpose_4(v2)
        v4 = torch.nn.functional.relu(v3)
        v5 = self.conv_transpose_5(v4)
        v6 = torch.nn.functional.relu(v5)
        v7 = self.conv_transpose_6(v6)
        v8 = torch.sigmoid(v7)
        v9 = v7 * v8
        v10 = self.conv_transpose_7(v9)
        v11 = torch.nn.functional.relu(v10)
        v12 = self.conv_transpose_8(v11)
        v13 = torch.sigmoid(v12)
        v14 = v12 * v13
        v15 = self.relu_32(v14)
        return v15
# Inputs to the model
x1 = torch.randn(1, 16, 28, 28)
