
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convtranspose1 = torch.nn.ConvTranspose1d(1, 4, 5, stride=1, padding=5)
    def forward(self, x1):
        v1 = self.convtranspose1(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 28)
