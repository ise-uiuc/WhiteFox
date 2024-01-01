
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(5, 1, 3)
        self.conv_t2 = torch.nn.ConvTranspose2d(1, 14, 2)
    def forward(self, x1):
        k1 = self.conv_t1(x1)
        k2 = k1 > 0
        k3 = k1 * 0.0573
        k4 = torch.where(k2, k1, k3)
        k5 = self.conv_t2(k4)
        return torch.tanh(k5)
# Inputs to the model
x1 = torch.randn(21, 5, 12, 4)
