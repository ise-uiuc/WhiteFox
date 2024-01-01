
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(9, 2, 9, stride=2, bias=False, padding=1)
        self.leaky_relu = torch.nn.LeakyReLU(self.conv_t.stride[0], True)
    def forward(self, x18):
        x19 = self.conv_t(x18)
        x20 = self.leaky_relu(x19)
        x21 = x20 * 0.90807
        x22 = x21 + 0.81302
        return torch.nn.functional.max_pool2d(x22, kernel_size=6, stride=2, padding=0)
# Inputs to the model
x18 = torch.randn(14, 9, 23, 7)
