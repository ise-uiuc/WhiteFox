
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convtranspose1 = torch.nn.ConvTranspose3d(16, 16, (1,2,2), stride=(2,2,2), padding=(1,0,0))
    def forward(self, x1):
        v1 = self.convtranspose1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 16, 32, 32, 32)
