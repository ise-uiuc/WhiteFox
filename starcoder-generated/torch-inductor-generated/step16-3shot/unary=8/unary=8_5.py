
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose4 = torch.nn.ConvTranspose1d(128, 64, 2, stride=1)
        self.conv_transpose3 = torch.nn.ConvTranspose1d(64, 32, 3, stride=1)
        self.conv_transpose2 = torch.nn.ConvTranspose1d(32, 64, 4, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose4(x1)
        v2 = self.conv_transpose3(v1)
        v3 = v2 + 3
        v4 = torch.clamp(v3, min=0)
        v5 = torch.clamp(v4, max=6)
        v6 = v2 * v5
        v7 = v6 / 6
        v8 = self.conv_transpose2(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 128, 100)
