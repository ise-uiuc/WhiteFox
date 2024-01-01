
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 1, 3, stride=1)
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = v1.transpose(3, 1) > 0
        v3 = v1.transpose(3, 1) * 0.125
        v4 = v1 * -1.0
        v5 = torch.where(v2, v3, v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 1, 1)
