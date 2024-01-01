
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tconv = torch.nn.ConvTranspose2d(8, 8, 3, stride=2, padding=1)
    def forward(self, x):
        x = self.tconv(x)
        x = torch.clamp(x, min=0., max=1.)
        return x
# Inputs to the model
x = torch.randn(1, 8, 64, 64)
