
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(16, 256, 5, stride=1, padding=2)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(64, 128, 3, stride=2, padding=1)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(128, 64, 2, stride=2, padding=0)
        self.conv_transpose4 = torch.nn.ConvTranspose2d(256, 32, 5, stride=1, padding=2)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = self.conv_transpose2(v1)
        v3 = self.conv_transpose3(v2)
        v4 = self.conv_transpose4(torch.cat((v3, v2, v1), 1))
        v5 = v4 * 0.5
        v6 = v4 * 0.7071067811865476
        v7 = torch.erf(v6)
        v8 = v7 + 1
        v9 = v5 * v8
        return v9
# Inputs to the model
x1 = torch.randn(1, 16, 128, 128)
