
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t_2 = torch.nn.ConvTranspose2d(480, 7, 2, stride=2)
    def forward(self, x1):
        v1 = self.conv_t_2(x1)
        v2 = v1 > 0
        v3 = v1 * 0.5
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(16, 480, 16, 16)
