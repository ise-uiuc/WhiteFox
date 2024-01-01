
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_6 = torch.nn.ConvTranspose2d(3, 3, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose_6(x1)
        v2 = torch.sigmoid(v1)
        v3 = torch.sigmoid(v2)
        v4 = torch.sigmoid(v3)
        v5 = torch.sigmoid(v4)
        v6 = torch.sigmoid(v5)
        v7 = v6 * v5
        v8 = v7 * v4
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
