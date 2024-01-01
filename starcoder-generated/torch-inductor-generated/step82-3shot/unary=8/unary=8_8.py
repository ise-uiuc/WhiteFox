
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(3, 3, 3, stride=1)
        self.conv_2 = torch.nn.Conv2d(3, 2, 3, stride=1)
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 2, 3, stride=1, padding=2)
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = self.conv_2(v1)
        v3 = self.conv_transpose(v2)
        v4 = v3 + 3
        v5 = torch.clamp(v4, min=0)
        v6 = torch.clamp(v5, max=6)
        v7 = v3 * v6
        v8 = v7 / 6
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
