
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convtranspose1 = torch.nn.ConvTranspose2d(4, 7, 2, stride=2, padding=2)
    def forward(self, x0):
        x1 = F.pad(x0, (2, 2, 2, 2))
        x2 = self.convtranspose1(x1)
        x3 = F.pad(torch.sigmoid(x2), pad=(2, 2, 2, 2))
        x4 = x2 * x3
        return x4
# Inputs to the model
x0 = torch.randn(1, 4, 5, 5)
