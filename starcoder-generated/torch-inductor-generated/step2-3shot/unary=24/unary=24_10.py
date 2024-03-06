
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convd = torch.nn.ConvTranspose2d(3,8,1,stride=1,padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 > 0
        v3 = v1 * 0.01
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)