
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 2, 3, stride=1, padding=1, dilation=3, groups=2, bias=0)
        self.clamp = torch.nn.Hardtanh()
        self.conv_add = torch.nn.Conv2d(2, 2, 3, stride=1, padding=(1, 2), dilation=1, groups=1, bias=0)
        self.conv_add_add = torch.nn.Conv2d(2, 2, 3, stride=1, padding=(2, 2), dilation=1, groups=1, bias=0)
        self.conv_div_div = torch.nn.Conv2d(2, 2, 3, stride=1, padding=(2, 3), dilation=1, groups=1, bias=0)
        self.conv_sub = torch.nn.Conv2d(2, 2, 3, stride=1, padding=4, dilation=1, groups=1, bias=0)
        self.conv_mul = torch.nn.Conv2d(2, 2, 3, stride=1, padding=(3, 4), dilation=1, groups=1, bias=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.clamp(v1)
        v3 = self.conv_add(v2)
        v3_out = self.conv_add_add(v3)
        v3_out = torch.div(v3_out, 3)
        v3_out = self.conv_div_div(v3_out)
        v3_out = torch.add(v3, 1)
        v3_out = self.conv_sub(v3_out)
        v3_out = torch.mul(v3, 2)
        v4 = self.conv_mul(v3)
        v4_out = (v4 * 4).relu()
        return v4_out + v3_out
# Inputs to the model
x1 = torch.randn(1, 2, 32, 32)
