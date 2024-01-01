
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 16, (3, 6), padding=(1, 1), stride=(1, 1))
        self.conv_transpose = torch.nn.ConvTranspose2d(16, 64, (5, 5), padding=(2, 2), stride=(1, 1))
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv_transpose(v2)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 4, 9)
