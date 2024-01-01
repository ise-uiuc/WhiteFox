
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(4, 16, 6, stride=5, padding=15)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 / 0.5000000000000001
        v3 = v1 / 0.7071067811865476
        v4 = torch.tanh(v3)
        v5 = v2.neg()
        v6 = torch.exp(v5)
        v6 = v2 * v4
        return v6
# Inputs to the model
x1 = torch.randn(1, 4, 45, 45)
