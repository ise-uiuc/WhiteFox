
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(2, 10, 3, stride=2, padding=0)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(10, 3, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        v6 = self.conv_transpose2(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 2, 28, 28)
