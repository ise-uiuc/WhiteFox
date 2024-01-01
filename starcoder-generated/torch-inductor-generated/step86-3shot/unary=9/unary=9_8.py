
class Model(torch.nn.Module):
    def __init__(self, padding1=1, padding2=2, padding3=3, dim=-1):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.pad1 = torch.nn.ZeroPad2d((padding1, padding1, padding1, padding1))
        self.pad2 = torch.nn.ZeroPad2d((padding2, padding2, padding2, padding2))
        self.pad3 = torch.nn.ZeroPad2d((padding3, padding3, padding3, padding3))
    def forward(self, x1):
        t1 = self.pad1(x1)
        t2 = self.pad2(x1)
        t3 = self.pad3(x1)
        v1 = self.conv(t1) + self.conv(t2) + self.conv(t3)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
