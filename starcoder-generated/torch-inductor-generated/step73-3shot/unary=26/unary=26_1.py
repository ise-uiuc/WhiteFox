
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(5, 55, 2, bias=False)
    def forward(self, x29):
        a1 = self.conv_t(x29)
        a2 = a1 > 0
        a3 = a1 * -0.994
        a4 = torch.where(a2, a1, a3)
        return a4
#Inputs to the model
x29 = torch.randn(8, 1, 50, 5, device='cpu')
