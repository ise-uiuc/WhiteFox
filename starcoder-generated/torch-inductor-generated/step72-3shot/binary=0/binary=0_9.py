
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pad = torch.nn.ConstantPad2d(20, 1.2)
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=2, padding=0)
        self.conv_out = torch.nn.Conv2d(3, 3, 1, stride=2, padding=0)
    def forward(self, x, other=1):
        v1 = self.pad(x)
        v2 = self.conv(v1)
        v3 = v2 + other
        v4 = self.conv_out(v3)
        return v4
# Inputs to the model
x = torch.randn(1, 3, 40, 40)
