
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convtranspose = torch.nn.ConvTranspose3d(6, 5, 1, stride=1, padding=0, output_padding=0)
    def forward(self, x1):
        v1 = self.convtranspose(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 6, 2, 3, 1)
