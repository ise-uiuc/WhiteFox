
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_0 = torch.nn.Conv2d(4, 4, 3, stride=2, padding=1, bias=False)
        self.relu_1 = torch.nn.ReLU()
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(4, 1, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.relu_3 = torch.nn.ReLU()
        self.conv_transpose_4 = torch.nn.ConvTranspose2d(1, 1, 3, stride=2, padding=1, output_padding=1)
    def forward(self, x1):
        v1 = torch.transpose(x1, -3, -1)
        v2 = torch.transpose(x1, -2, -1)
        v3 = v1[:]
        v4 = self.conv_transpose_0(v3)
        v5 = v2[:]
        v6 = self.relu_1(v4)
        v7 = self.conv_transpose_2(v5)
        v8 = v3[:]
        v9 = self.relu_3(v7)
        v10 = self.conv_transpose_4(v8)
        v11 = torch.transpose(v10, -3, -1)
        v12 = torch.transpose(v11, -2, -1)
        v13 = torch.transpose(v12, -1, -3)
        return v13
# Inputs to the model
x1 = torch.randn(1, 4, 6, 6)
