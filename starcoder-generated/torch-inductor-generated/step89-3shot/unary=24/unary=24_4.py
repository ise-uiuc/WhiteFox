
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(43, 78, 3, stride=1, padding=1, output_padding=1)
        self.conv2 = torch.nn.ConvTranspose2d(78, 16, 2, stride=2, padding=1, dilation=2)
        self.conv3 = torch.nn.ConvTranspose2d(16, 12, 3, stride=3, padding=2, output_padding=1)
        self.conv4 = torch.nn.ConvTranspose2d(12, 82, 1, stride=1, padding=0)
    def forward(self, x):
        negative_slope = -0.76766074
        v1 = self.conv1(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        v5 = self.conv2(v4)
        v6 = v5 > 0
        v7 = v5 * negative_slope
        v8 = torch.where(v6, v5, v7)
        v9 = self.conv3(v8)
        v10 = v9 > 0
        v11 = v9 * negative_slope
        v12 = torch.where(v10, v9, v11)
        v13 = self.conv4(v12)
        v14 = v13 > 0
        v15 = v13 * negative_slope
        v16 = torch.where(v14, v13, v15)
        return v16
# Inputs to the model
x1 = torch.randn(91, 43, 13, 12)
