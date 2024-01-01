
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gelu = torch.nn.GELU()
        self.conv = torch.nn.Conv2d(10, 10, 3, stride=2, padding=[[3, 5], [4, 6]], dilation=2)
        self.conv_transpose = torch.nn.ConvTranspose2d(10, 10, 3, stride=2, padding=[[7, 9], [10, 12]], dilation=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv_transpose(v1)
        v3 = v2 - 0.4
        v4 = self.gelu(v3)
        v5 = v4 + 0.1
        return v4 + v5
# Input to the model
x1 = torch.randn(3, 10, 8, 8)
# Model end
