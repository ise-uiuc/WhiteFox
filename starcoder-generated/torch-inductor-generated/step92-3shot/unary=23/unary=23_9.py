
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convtranspose = torch.nn.ConvTranspose3d(8, 5, 2, stride=1, bias=False)
    def forward(self, x1):
        v1 = self.convtranspose(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 8, 5, 3, 2)
