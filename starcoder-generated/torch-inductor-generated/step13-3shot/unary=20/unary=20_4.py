
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convT = torch.nn.ConvTranspose3d(in_channels=1, out_channels=1, kernel_size=3, stride=3, padding=3)
    def forward(self, x1):
        v1 = self.convT(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 303, 504, 91)
