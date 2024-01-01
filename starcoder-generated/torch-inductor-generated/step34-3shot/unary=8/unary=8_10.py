
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(5, 10, 2, stride=2, padding=1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(10, 3, 2, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = v1 + 3
        v3 = torch.clamp(v2, min=0)
        v4 = torch.clamp(v3, max=6)
        v5 = v1 * v4
        v6 = v5 / 6
        y1 = self.conv_transpose2(v6)
        return y1
# Inputs to the model
x1 = torch.randn(1, 5, 224, 224)
