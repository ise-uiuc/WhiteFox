
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose2 = torch.nn.ConvTranspose2d(3, 3, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose2(x1)
        v2 = torch.sigmoid(v1)
        v3 = v2 + v2
        v4 = v3 + v3
        v5 = v4 + v4
        v6 = v5 + v5
        v7 = v6 + v6
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
