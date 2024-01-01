
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 6, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1.add(3)
        v3 = F.clamp(v2, 0, 6)
        v4 = torch.div(v3, 6)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
