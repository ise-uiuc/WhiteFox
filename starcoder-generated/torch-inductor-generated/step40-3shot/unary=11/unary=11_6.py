
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(16, 32, 5, stride=1, padding=1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(8, 16, 3, stride=1, padding=1)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(4, 8, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.conv_transpose2(v1)
        v3 = self.conv_transpose3(v2)
        v4 = v3 + 4
        v5 = torch.clamp_min(v4, 0)
        v6 = torch.clamp_max(v5, 8)
        v7 = v6 / 8
        return v7, v3, v2, v1

# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
