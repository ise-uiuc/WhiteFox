
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 1, 3, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.clamp(v1, 0, 16)
        v3 = v2 / 16
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
