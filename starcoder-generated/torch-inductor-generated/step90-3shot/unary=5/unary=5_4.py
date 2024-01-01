
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(1, 1, 1, stride=1, padding=0)
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 1, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv2d(x1)
        v2 = self.conv_transpose(v1)
        v3 = v2 * 0.5
        v4 = v2 * 0.7071067811865476
        v5 = torch.erf(v4)
        v6 = v5 + 1
        v7 = v3 * v6
        return v7
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
