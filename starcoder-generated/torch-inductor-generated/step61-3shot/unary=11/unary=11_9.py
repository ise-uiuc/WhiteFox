
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 2, kernel_size=(6, 6), stride=(1, 1), padding=(1, 1))
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        return v5
# Inputs to the model
x1 = torch.randn(64, 8, 64, 64)
