
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.7071067811865476
        v3 = torch.max(x1, v2)
        v4 = self.conv(v3)
        v5 = v4 * 0.7071067811865476
        v6 = torch.max(x1, v5)
        v7 = self.conv(v6)
        v8 = v7 * 0.7071067811865476
        v9 = torch.max(x1, v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 119, 154)
