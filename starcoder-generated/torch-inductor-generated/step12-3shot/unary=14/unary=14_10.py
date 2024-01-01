
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(7, 8, 2, stride=2, padding=1)
        self.sigmoid_1 = torch.nn.Sigmoid()
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(8, 6, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = self.sigmoid_1(v1)
        v3 = self.conv_transpose_2(v1)
        v4 = self.sigmoid_1(v3)
        v5 = v4 + v2
        return v5
# Inputs to the model
x1 = torch.randn(1, 7, 64, 64)
