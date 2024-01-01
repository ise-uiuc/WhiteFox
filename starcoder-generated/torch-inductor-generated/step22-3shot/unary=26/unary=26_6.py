
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(9, 1, 2, stride=(2, 1), bias=False)
        self.n = 4
    def forward(self, x2):
        v1 = self.conv_t(x2)
        v2 = v1 > 0
        v3 = v1 * 0
        v4 = torch.where(v2, v1, v3)
        v5 = -torch.sum(v4) * self.n
        return v5
# Inputs to the model
x2 = torch.randn(1, 9, 8, 5)
