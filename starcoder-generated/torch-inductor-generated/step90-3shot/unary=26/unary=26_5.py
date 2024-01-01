
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose3d(63, 63, 5, stride=2, bias=True)
    def forward(self, x):
        i1 = self.conv_t(x)
        i2 = i1 > 0
        i3 = i1 * -0.6060502753448486
        i4 = torch.where(i2, i1, i3)
        return torch.nn.functional.tanh(i4)
# Inputs to the model
x = torch.randn(1, 63, 48, 22, 30, device='cpu')
