
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(45, 12, stride=2)
    def forward(self, x2):
        v5 = self.conv_t(x2)
        v6 = v5 > 0
        v7 = v5 * 0.1472
        v8 = torch.where(v6, v5, v7)
        return v8
# Inputs to the model
x2 = torch.randn(40, 45, 5, 5)
