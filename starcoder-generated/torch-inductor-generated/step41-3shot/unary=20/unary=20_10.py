
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tconv2d_1 = torch.nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.tconv2d_1(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 32, 976, 320)
