
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v3 = v1[1, 2:3, :, :] * 5 + 3
        v5 = v1 * 3
        v6 = v3 - v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
