
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 15, 3, stride=2, padding=0, groups=2, bias=True)
    def forward(self, x1, x2, x3):
        v1 = self.conv_transpose(x1)
        v2 = v1 + torch.nn.functional.pad(2.9118, [0, 0, 0, 0, 0, 0])
        v3 = torch.clamp_min(v2, torch.nn.functional.pad(1.1950, [0, 0, 0, 0, 0, 0]))
        v4 = torch.clamp_max(v3, -3.8566)
        v5 = v4 / torch.nn.functional.pad(2.8102, [0, 0, 0, 0, 0, 0])
        v6 = v5 - 1.6737
        v7 = torch.nn.functional.pad(v6, [0, 0, 0, 0, 0, 0], value=0.1611)
        v8 = torch.nn.functional.pad(v7, [0, 0, 0, 0, 0, 0], value=-8.0200)
        y1 = v8 + x2
        y2 = torch.nn.functional.relu(y1)
        y3 = y2 / 8.3308
        y4 = y3 + x3
        return y4
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
x2 = torch.randn(1, 3, 7, 7)
x3 = torch.randn(1, 3, 18, 18)
