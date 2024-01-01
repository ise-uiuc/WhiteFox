
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 3, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = (v1 + 3)*2
        v3 = v2 - 1
        v4 = v3 / 8
        return v4
# Inputs to the model
x1 = torch.randn(3, 5, 6, 7)
