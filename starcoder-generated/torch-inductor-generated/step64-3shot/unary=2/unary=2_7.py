
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pad = torch.nn.ZeroPad2d((0, 0, 0, 2))
        self.conv = torch.nn.Conv2d(4, 1, (1, 1), stride=(1, 1))
        self.pad_1 = torch.nn.ZeroPad2d((2, 2, 2, 0))
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 8, (2, 2), stride=(2, 2), dilation=(1, 1), padding=(2, 2))
        self.pad_2 = torch.nn.ZeroPad2d((0, 0, 0, 2))
        self.conv_1 = torch.nn.Conv2d(8, 4, (1, 1), stride=(1, 1), dilation=(2, 2), padding=(2, 2))
    def forward(self, x1):
        v1 = self.pad(x1)
        v2 = self.conv(v1)
        v3 = v2.permute((0, 3, 2, 1))
        v4 = self.pad_1(v3)
        v5 = self.conv_transpose(v4)
        v6 = v5.permute((0, 3, 2, 1))
        v7 = self.pad_2(v6)
        v8 = self.conv_1(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 4, 6, 6)
model = Model()
