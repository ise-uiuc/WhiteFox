
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(3, 16, 2, stride=1, padding=0, bias=False)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(16, 32, 3, stride=1, padding=0, bias=False)
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(32, 1, 2, stride=0, padding=0, bias=False)
        self.conv2d_1 = torch.nn.Conv2d(1, 1, 3, stride=3, padding=0, bias=False)
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = self.conv_transpose_2(v1)
        v3 = self.conv_transpose_3(v2)
        v4 = self.conv2d_1(x1)
        v5 = v3 + v4
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 12, 12)
