
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose1d(19, 19, 15, 4, groups=19)
    def forward(self, x):
        y = self.conv(x)
        pass
# Inputs to the model
x = torch.randn(5, 19, 26)
