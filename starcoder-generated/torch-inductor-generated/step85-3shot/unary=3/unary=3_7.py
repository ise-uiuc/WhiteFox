
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(20, 59, (1, 2), stride=(1, 2), padding=(0, 1))
        self.conv2 = torch.nn.Conv2d(59, 85, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = v6.reshape(v6.size(0), v6.size(1), -1)
        v8 = self.conv2(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 20, 171, 157)
