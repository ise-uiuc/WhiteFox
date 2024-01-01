
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 10, 3)
        self.conv_transpose = torch.nn.ConvTranspose2d(10, 3, 7, stride=1, padding=3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv_transpose(v1)
        v3 = v2 * 0.699975
        v4 = v2 * v2 * v2
        v5 = v2 * 0.00051801
        v6 = torch.asin(v5)
        v7 = torch.sqrt(v4)
        v8 = v6 - v7
        v9 = v3 * v8
        return v9
# Inputs to the model
x1 = torch.randn(12, 3, 15, 9)
