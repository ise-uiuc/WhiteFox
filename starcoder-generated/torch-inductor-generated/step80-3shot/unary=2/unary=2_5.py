
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(256, 512, 4, groups=4, bias=False)
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(1280, 512, 1, bias=False)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.conv = torch.nn.Conv2d(512, 512, 1, stride=1, bias=False)
        self.conv_1 = torch.nn.Conv2d(256, 256, 1, stride=1, bias=False)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.conv_transpose_1(v1)
        v3 = self.dropout(v2)
        v4 = self.conv(v3)
        v5 = self.conv_1(x1)
        v6 = v5 + v4
        return v6
# Inputs to the model
x1 = torch.randn(1, 256, 3, 5)
