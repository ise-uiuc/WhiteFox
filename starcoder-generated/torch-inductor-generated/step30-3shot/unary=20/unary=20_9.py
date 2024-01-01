
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose1d(1, 2, kernel_size=2)
        self.conv_t_2 = torch.nn.ConvTranspose2d(1, 2, kernel_size=2)
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = self.conv_t_2(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 10)
