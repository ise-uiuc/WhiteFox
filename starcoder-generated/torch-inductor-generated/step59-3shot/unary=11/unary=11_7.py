
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(3, 8, 3, stride=2, padding=0)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(8, 8, 3, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = v1 * 3
        v3 = self.conv_transpose2(v2)
        v4 = v3 + 8
        v5 = v4 / 6
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
