
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 64, 2, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.flatten(v1, 1)
        v3 = torch.matmul(v2, v2)
        v4 = torch.flatten(v3, 1)
        v5 = v2 * v4
        v6 = v5 / 6
        return v6
# Inputs to the model
x1 = torch.randn(1, 1, 22, 33)
