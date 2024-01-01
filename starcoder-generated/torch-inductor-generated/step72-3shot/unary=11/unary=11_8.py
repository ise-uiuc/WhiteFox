
class Model(torch.nn.Module):
    def __init__(self, batch_size: int, channels: int):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(64, 128, 55, stride=(channels, 1), padding=66, dilation=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        return v5
# Inputs to the model
x1 = torch.randn(1000000000, 120, 100, 999)
