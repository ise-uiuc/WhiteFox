
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(3, 3, 2)
    def forward(self, x):
        x = self.conv_t(x)
        return x
# Inputs to the model
x = torch.randn(3, 2, 3, 2)
