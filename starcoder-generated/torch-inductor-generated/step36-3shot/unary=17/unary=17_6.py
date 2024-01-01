
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convtrans_layer = torch.nn.ConvTranspose2d(3, 16, 3, stride=1, padding=1)
    def forward(self, x):
        v1 = self.convtrans_layer(x)
        return v1
# Inputs to the model
x = torch.randn(1, 3, 128, 128)
