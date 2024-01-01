
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(44, 35, 3, stride=2, padding=3)
        self.conv2 = torch.nn.Conv2d(35, 36, 5, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(36, 33, 5, stride=2, padding=2)
        self.conv4 = torch.nn.Conv2d(33, 41, 5, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v2 * 0.5
        v4 = v2 * 0.7071067811865476
        v5 = torch.erf(v4)
        v6 = v5 + 1
        v7 = self.conv3(v6)
        v8 = self.conv4(v7)
        return v8
# Inputs to the model
x1 = torch.randn(15, 44, 94, 94)
