
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convtranspose = torch.nn.ConvTranspose2d(43, 32, 3, stride=2)
    def forward(self, x1):
        v1 = self.convtranspose(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 43, 64, 64)
