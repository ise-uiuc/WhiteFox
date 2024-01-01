
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(1, 1, kernel_size=(2, 3))
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.nn.ReLU()(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)
