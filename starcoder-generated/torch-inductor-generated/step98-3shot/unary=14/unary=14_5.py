
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(4, 4, 3, stride=1, padding=1)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(4, 4, 5, stride=1, padding=0)
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(4, 4, 3, stride=1, padding=2, dilation=1)
        self.conv_transpose_4 = torch.nn.ConvTranspose2d(4, 1, 3, stride=1, padding=1)
    def forward(self, x1):
        g1 = self.conv_transpose_1(x1)
        g2 = torch.sigmoid(g1)
        g3 = g1 * g2
        g4 = self.conv_transpose_2(g3)
        g5 = torch.sigmoid(g4)
        g6 = g4 * g5
        g7 = self.conv_transpose_3(g6)
        g8 = torch.sigmoid(g7)
        g9 = g7 * g8
        g10 = self.conv_transpose_4(g9)
        g11 = torch.sigmoid(g10)
        g12 = g10 * g11
        return g12
# Inputs to the model
x1 = torch.randn(1, 4, 16, 16)
