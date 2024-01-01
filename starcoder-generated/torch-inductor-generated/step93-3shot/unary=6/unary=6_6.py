
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv11 = torch.nn.Conv2d(13, 13, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(13)
        self.conv12 = torch.nn.Conv2d(13, 13, 1, stride=1, padding=1)
        self.conv21 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        t1 = self.conv11(x1)
        t2 = self.bn1(t1)
        t3 = self.conv12(t2)
        v1 = self.conv21(x1)
        v2 = v1 + 3
        v3 = torch.nn.functional.relu6(v2)
        v4 = v1 * v3
        t4 = t3 * t4
        t5 = t3 + 3
        t6 = torch.nn.functional.relu6(t5)
        t7 = self.bn1(t6)
        t8 = t1 * t7
        v5 = v4 / 6
        v6 = t8 - t3
        v7 = torch.clamp_max(v6, 6)
        v8 = torch.clamp_min(v7, 0)
        v9 = self.bn1(v8)
        v10 = v5 + v9
        return v10
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
