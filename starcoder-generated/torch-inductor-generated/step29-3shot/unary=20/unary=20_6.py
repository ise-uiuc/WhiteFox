
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(3, 3, kernel_size=1)
    def forward(self, x):
        t = torch.conv_t(x)
        v = torch.sigmoid(t)
        return t
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
