
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_0 = torch.nn.ConvTranspose2d(1, 192, 11, stride=4, padding=2)
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(192, 161, 5, stride=2, padding=2)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(161, 96, 5, stride=2, padding=2)
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(96, 3, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose_0(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv_transpose_1(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.conv_transpose_2(v4)
        v6 = torch.sigmoid(v5)
        v7 = self.conv_transpose_3(v6)
        v8 = torch.sigmoid(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 1, 100, 100)
