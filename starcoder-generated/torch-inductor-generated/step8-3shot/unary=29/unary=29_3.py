
class Model(nn.Module):
    def __init__(self, min_value=-2.6050181, max_value=5.085745604477906):
        super().__init__()
        # 7x7 conv2d
        self.pad = nn.ConstantPad2d(padding=1, value=0.0)
        self.conv2d = nn.Conv2d(4, 144, (7, 7), stride=1, bias=False, padding=0)
        # transpose Conv2D
        self.convt = nn.ConvTranspose2d(144, 144, 2, stride=2, bias=False)
        self.bn = nn.GroupNorm(144*2, 144, eps=1.8832713901085855e-05)
        self.act = nn.GELU()
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, inputs):
        out1 = self.pad(inputs)
        out2 = self.conv2d(out1)
        out3 = torch.clamp_min(out2, self.min_value)
        out4 = torch.clamp_max(out3, self.max_value)
        out5 = self.convt(out4)
        out6 = torch.clamp_min(out5, self.min_value)
        out7 = torch.clamp_max(out6, self.max_value)
        out8 = self.bn(out7)
        out9 = self.act(out8)
        return out9

# Inputs to the model
x1 = torch.randn(1, 4, 32, 32)
