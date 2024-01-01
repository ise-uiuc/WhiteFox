
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 10, 9, stride=1, padding=11)
        self.tconv1 = torch.nn.ConvTranspose2d(10, 1, 23, size=None, stride=1, padding=7)
        self.tconv2 = torch.nn.ConvTranspose2d(28, 76, 11, size=(22, 22), stride=1, padding=8)
        self.tconv3 = torch.nn.ConvTranspose2d(68, 1, 9, size=(22, 22), stride=1, padding=10)
    def forward(self, x6):
        v1 = self.conv(x6)
        v2 = v1 * 0.5
        v3 = v1 * v1
        v4 = v3 * v1
        v5 = v4 * 0.044715
        v6 = v1 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v2 * v9
        v11 = self.tconv1(v10)
        v12 = self.tconv2(v8)
        v13 = self.tconv3(v12)
        v14 = v11 * v13
        return v14
# Inputs to the model
x6 = torch.randn(1, 2, 32, 32)
