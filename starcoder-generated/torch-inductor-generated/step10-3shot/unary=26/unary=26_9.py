
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=0)
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=1)
        self.addm = torch.nn.ReLU()
    def forward(self, x1):
        x2 = self.conv_transpose_2(x1)
        x3 = self.conv_transpose_1(x1)
        x4 = x2 + x3
        x5 = x4 > 0
        x6 = self.addm(x4)
        x7 = torch.where(x5, x4, x6)
        return x7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
