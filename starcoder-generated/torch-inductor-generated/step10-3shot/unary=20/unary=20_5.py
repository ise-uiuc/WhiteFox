
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(3, 2, kernel_size=3, stride=1)
    def forward(self, x):
        a1 = self.conv_t(x)
        a1 = torch.sigmoid(a1)
        return a1
# Inputs to the model
x = torch.randn(1, 3, 128, 128)
