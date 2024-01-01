
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(29, 26, 4, padding=2, stride=2)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(26, 27, 1, padding=0, stride=1)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(27, 28, 1, padding=0, stride=1)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = self.conv_transpose2(v1)
        v3 = v2 * 0.5
        v4 = v2 * v2 * v2
        v5 = v4 * 3.9774841810124673e-05
        v6 = v2 + v5
        v7 = torch.tanh(v6)
        v8 = v7 + 1
        v9 = v3 * v8
        v10 = self.conv_transpose3(v9)
        return v10
# Inputs to the model
x1 = torch.randn(12, 29, 3, 7)
