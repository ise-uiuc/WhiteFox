
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(6, 4, 15, stride=1, padding=7)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(4, 3, 3, stride=1, padding=1)
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(3, 3, 9, stride=1, padding=4)
        self.conv_transpose_4 = torch.nn.ConvTranspose2d(3, 1, 1, stride=2, padding=0)
    def forward(self, x1):
        w1 = self.conv_transpose_1(x1)
        w2 = torch.sigmoid(w1)
        w3 = w1 * w2
        w4 = self.conv_transpose_2(w3)
        w5 = torch.sigmoid(w4)
        w6 = w4 * w5
        w7 = self.conv_transpose_3(w6)
        w8 = torch.sigmoid(w7)
        w9 = w7 * w8
        w10 = self.conv_transpose_4(w9)
        return w10
# Inputs to the model
x1 = torch.randn(1, 6, 6, 6)
