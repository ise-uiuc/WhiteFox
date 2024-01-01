
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        conv1 = torch.nn.ModuleList()
        for i in range(2):
            conv1.append(torch.nn.Conv2d(3, 8, 1, stride=1, padding=1))
        self.conv1 = conv1
        self.conv2 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        r1 = self.conv1[0](x1)
        r2 = self.conv1[1](x1)
        v1 = r1 + r2
        v2 = torch.clamp(v1, 0, 6)
        v3 = r1 * v2
        v4 = v3 / 6
        v5 = self.conv2(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
