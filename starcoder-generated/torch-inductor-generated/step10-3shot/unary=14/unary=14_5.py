
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convtranspose1 = torch.nn.ConvTranspose1d(64, 512, kernel_size=3, stride=1, padding=1,)
        self.convtranspose2 = torch.nn.ConvTranspose2d(64, 512, 3, stride=1, padding=1,)
    def forward(self, x1):
        v1 = self.convtranspose1(x1)
        v2 = self.convtranspose2(x1)
        v3 = torch.mean(torch.reshape(v1, (-1, 512, 1)))
        v4 = torch.sigmoid(v3)
        return v4 + v2
# Inputs to the model
x1 = torch.randn(1, 64, 64)
