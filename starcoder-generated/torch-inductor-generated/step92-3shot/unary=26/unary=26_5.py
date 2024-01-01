
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(235, 405, 3, stride=1, padding=2, bias=False)
    def forward(self, x9):
        v1 = self.conv_t(x9)
        v2 = v1 > 0
        v3 = v1 * -0.901389
        v4 = torch.where(v2, v1, v3)
        return torch.nn.functional.leaky_relu(v4)
# Inputs to the model
x9 = torch.randn(5, 235, 23, 32)
