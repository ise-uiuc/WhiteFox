
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_block = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(2, 4, 3, stride=2, padding=0),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(4, 3, (1, 1)),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(3, 2, 5, stride=4, padding=2)
        )
    def forward(self, x1):
        v1 = self.conv_transpose_block(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        return v5
# Inputs to the model
x1 = torch.randn(1, 2, 25, 25)
