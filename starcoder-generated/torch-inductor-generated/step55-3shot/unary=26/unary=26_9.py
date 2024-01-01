
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose1d(5, 8, 7, stride=5)
        self.conv = torch.nn.Conv1d(8, 12, 3, dilation=2)
    def forward(self, x1):
        x2 = self.conv_t(x1)
        x3 = x1 > 0
        x4 = x1 * 0.8
        x5 = torch.where(x3, x1, x4)
        x6 = self.conv(x5)
        return x6
# Inputs to the model
x1 = torch.randn(13, 5, 10)
