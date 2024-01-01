
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convtranspose3 = torch.nn.ConvTranspose3d(1, out_channels=1, kernel_size=28, stride=3, padding=7)
    def forward(self, x1):
        v1 = self.convtranspose3(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 224, 224)
