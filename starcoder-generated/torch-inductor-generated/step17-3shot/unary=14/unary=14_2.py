
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv15_1 = torch.nn.ConvTranspose2d(11, 11, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv15_1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 11, 64, 64)
