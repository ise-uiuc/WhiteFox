
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 3, (5, 7), stride=(2, 1))
    def forward(self, x1, x2, other=1, padding1=None):
        x1 = F.conv2d(x1, self.conv.weight, self.conv.bias, (1, 1), (2, 0), (0, 0), (2, 1))
        x2 = self.conv(x2)
        x3 = x1 * 3 + x2 / other
        x3[:, :, 11:13, :] = x3[:, :, 11:13, :] + padding1
        return x3
# Inputs to the model
x1 = torch.randn(1, 2, 64, 64)
x2 = torch.randn(1, 2, 100, 150)
