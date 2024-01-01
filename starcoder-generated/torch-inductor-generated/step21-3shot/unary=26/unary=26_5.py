
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 3, 1, stride=1, padding=1)
        self.pooling = torch.nn.AvgPool2d(2, stride=2)
    def forward(self, x4):
        x1 = self.conv_t(x4)
        x7 = self.pooling(x1)
        x6 = x1 > 0
        x3 = x1 * 0.7
        x5 = torch.where(x6, x1, x3)
        x8 = torch.tanh(x5)
        x9 = torch.cat((x7, x5), 1)
        return x9

# Inputs to the model
x4 = torch.randn(8, 1, 16, 16)
