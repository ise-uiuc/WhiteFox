
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convtranspose3 = torch.nn.ConvTranspose2d(4, 3, 1, stride=3, padding=7)
    def forward(self, x1):
        v1 = self.convtranspose3(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 4, 35, 25)
