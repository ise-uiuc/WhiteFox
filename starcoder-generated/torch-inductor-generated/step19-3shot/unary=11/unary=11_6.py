
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(64, 32, 2, stride=2, padding=1)
    def forward(self, x1):
        v1 = torch.tanh(x1)
        v2 = self.conv_transpose(v1)
        v3 = v2 + 3
        v4 = torch.clamp_min(v3, 0)
        v5 = torch.clamp_max(v4, 6)
        v6 = v5 / 6
        w1 = torch.tanh(v6)
        return w1
# Inputs to the model
x1 = torch.randn(1, 64, 28, 28)
