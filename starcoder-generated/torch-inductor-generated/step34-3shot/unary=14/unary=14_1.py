
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 1, 2, stride=2, padding=0)
    def forward(self, x2):
        v2 = self.conv_transpose(x2)
        v4 = torch.sigmoid(v2)
        v5 = v2 * v4
        return v5
# Inputs to the model
x2 = torch.randn(1, 3, 64, 64)
