
class Model(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose3d(dim, 1, 3, stride=1)
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = v1 > 0
        v3 = v1 * -0.75
        v4 = torch.where(v2, v1, v3)
        return v4
dim = 1
# Inputs to the model
x1 = torch.randn(1, 1, 1, 1, 1)
