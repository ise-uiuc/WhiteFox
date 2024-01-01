
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(40, 6, kernel_size=(5, 3), bias=False)
    def forward(self, x103):
        x1 = self.conv_t(x103)
        x2 = x1 > 0
        x3 = x1 * -0.783
        x4 = torch.where(x2, x1, x3)
        return torch.nn.functional.interpolate(torch.tanh(x4), (26, 42))
# Inputs to the model
x103 = torch.randn(16, 40, 23, 6, affine=True, device='cpu', requires_grad='True')
