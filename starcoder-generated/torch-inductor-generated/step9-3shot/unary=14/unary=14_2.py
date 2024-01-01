
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transposeconv1d = torch.nn.ConvTranspose1d(2, 4, 2, stride=2)
    def forward(self, x1):
        v1 = self.transposeconv1d(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(3, 2, 3)
