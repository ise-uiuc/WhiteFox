
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 64, 2, stride=2, padding=1)
        self.gelu_impl = torch.nn.GELU()
    def forward(self, x3):
        x4 = self.conv_t(x3)
        x5 = self.gelu_impl(x4)
        return x5
# Inputs to the model
x3 = torch.randn(1,1,1,256)
