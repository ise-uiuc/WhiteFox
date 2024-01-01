
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=2)
        self.conv_transpose = torch.nn.ConvTranspose2d(16, 8, 3, stride=1, padding=2)
        self.batch_norm = torch.nn.BatchNorm2d(16)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.conv_transpose(v1)
        v3 = self.batch_norm(v2)
        v4 = v3 + 3
        v5 = torch.clamp_min(v4, 0)
        v6 = torch.clamp_max(v5, 6)
        v7 = v6 / 6
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
