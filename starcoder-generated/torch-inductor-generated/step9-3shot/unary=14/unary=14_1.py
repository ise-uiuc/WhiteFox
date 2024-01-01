
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convtranspose2d = torch.nn.ConvTranspose2d(1, 1, kernel_size=3, output_padding=1)
    def forward(self, x1):
        v1 = self.convtranspose2d(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 2, 3)
