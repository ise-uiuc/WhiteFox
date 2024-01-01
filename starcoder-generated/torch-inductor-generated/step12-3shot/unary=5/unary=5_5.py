
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(1, 4, 5, stride=2)
        self.conv_transpose = torch.nn.ConvTranspose2d(4, 1, 5, stride=1, padding=5)
    def forward(self, x1):
        t1 = self.conv2d(x1)
        v1 = self.conv_transpose(t1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 1, 16, 16)
