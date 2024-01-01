
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv10 = torch.nn.Conv2d(513, 604, 3, stride=1, padding=1, groups=1)
        self.conv20 = torch.nn.Conv2d(604, 210, 1, stride=1, padding=0, groups=1)
        self.conv30 = torch.nn.Conv2d(210, 363, 1, stride=1, padding=0, groups=1)
    def forward(self, x1, x2):
        v1 = self.conv10(x1)
        v2 = self.conv20(v1)
        v3 = self.conv30(v2)
        v4 = torch.cat([v3, x2], 1)
        v5 = v4.sum(1, keepdim=True)
        v6 = torch.cat([v4, v5], 1)
        v7 = v6[:, :, 0, 0] + 27.029761505126953
        return v7
# Inputs to the model
x1 = torch.randn(1, 513, 1, 1)
x2 = torch.randn(1, 70056, 1, 1)
