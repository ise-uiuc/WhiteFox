
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(16, 2, 7, stride=2, padding=3)
        self.conv_2 = torch.nn.Conv2d(12, 16, 7, stride=2, padding=3)
        self.conv_3 = torch.nn.Conv2d(16, 12, 7, stride=2, padding=3)
    def forward(self, x1):
        v1 = self.conv_transpose_2(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_2(v3)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        v7 = self.conv_3(v6)
        v8 = torch.sigmoid(v7)
        v9 = v7 * v8
        return v9
# Inputs to the model
x1 = torch.randn(1, 2, 14, 14)
