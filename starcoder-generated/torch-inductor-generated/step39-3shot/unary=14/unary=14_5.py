
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_0 = torch.nn.Conv2d(221, 32, 3, stride=2, padding=1, dilation=1)
        self.conv_transpose_1 = torch.nn.ConvTranspose3d(32, 121, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv2d_0(x1)
        t1 = v1
        v2 = self.conv_transpose_1(t1)
        v3 = torch.sigmoid(v2)
        v4 = v2 * v3
        v5 = torch.sigmoid(v3)
        v7 = v5 * v4
        v8 = torch.clamp(v7, 0.0, 6.0) 
        return v8
# Inputs to the model
x1 = torch.randn(1, 221, 147, 147)
