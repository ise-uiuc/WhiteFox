
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(11, 8, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(8, 11, 3, stride=1, padding='same')
        self.conv3 = torch.nn.Conv2d(8, 11, 3, stride=1, padding=None)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = self.conv2(v3) * 0.5
        v5 = torch.erf(v3)
        v6 = v5 + 1
        v8 = self.conv3(v5) * 0.5
        v7 = torch.erf(v3)
        v10 = v7 + 1
        return v8
# Inputs to the model
x1 = torch.randn(1, 11, 63, 24)
