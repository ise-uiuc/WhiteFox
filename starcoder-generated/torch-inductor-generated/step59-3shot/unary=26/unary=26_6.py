
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(35, 966, 1, bias=True)
    def forward(self, x1):
        a1 = self.conv_t(x1)
        a2 = a1 > 0
        a3 = a1 * 0.39
        a4 = torch.where(a2, a1, a3)
        return torch.nn.functional.softplus(torch.nn.functional.relu(a4))
# Inputs to the model
x1 = torch.randn(4, 35, 9, 3)
