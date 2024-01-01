
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv = torch.nn.ConvTranspose2d(8, 8, kernel_size=9, stride=9, padding=9)
    def forward(self, x):
        out = self.deconv(x)
        out = torch.sigmoid(out)
        return out
# Inputs to the model
x= torch.randn(1, 8, 1, 1)
