
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose3d(1, 32, 2, stride=(1, 2, 2), padding=1, bias=True)
    def forward(self, x):
        y = self.conv_t(x)
        return y
# Inputs to the model
x = torch.randn(2, 1, 4, 5, 6)
