
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_5 = torch.nn.ConvTranspose2d(167, 70, 1, stride=1, padding=0)
        self.conv2d_1 = torch.nn.Conv2d(70, 35, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose_5(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v5 = self.conv2d_1(v3) # 35
        v6 = torch.sigmoid(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 167, 6, 6)
