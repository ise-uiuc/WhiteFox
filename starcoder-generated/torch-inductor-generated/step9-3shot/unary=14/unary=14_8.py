
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid6 = torch.nn.Sigmoid()
        self.transposeconv1d8 = torch.nn.ConvTranspose1d(2, 1, kernel_size=(1,), stride=(1,))
    def forward(self, x1):
        v1 = self.transposeconv1d8(x1)
        v2 = self.sigmoid6(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 3)
