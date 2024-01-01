
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv01 = torch.nn.Conv2d(3, 32, 1, stride=2, padding=1)
        self.conv02 = torch.nn.Conv2d(32, 16, 1, stride=2, padding=1)
        self.conv03 = torch.nn.Conv2d(16, 32, 1, stride=1, padding=1)
        self.conv04 = torch.nn.Conv2d(32, 16, 1, stride=1, padding=1)
        self.conv05 = torch.nn.Conv2d(16, 16, 3, stride=2, padding=1)
        self.conv06 = torch.nn.Conv2d(16, 4, 1, stride=2, padding=1)
        self.conv07 = torch.nn.Conv2d(4, 16, 2, stride=2, padding=1)
        self.conv08 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv09 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv01(x1)
        v2 = self.conv02(v1)
        v3 = self.conv03(v2)
        v4 = self.conv04(v1)
        v5 = self.conv05(v4)
        v6 = self.conv06(v5)
        v7 = self.conv07(v5)
        v8 = self.conv08(v6)
        v9 = self.conv09(v7)
        v10 = v8 + v9
        return v10
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
