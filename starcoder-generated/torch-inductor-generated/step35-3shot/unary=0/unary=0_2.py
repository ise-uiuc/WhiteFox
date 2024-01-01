
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 2, 3, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v1 * v1
        v4 = torch.reshape(v3, (1, 8, 32, 48))
        v5 = v4 * v1
        v6 = v5 * 0.044715
        v7 = v1 + v6
        v8 = v7 * 0.7978845608028654
        v9 = torch.sigmoid(v8)
        v10 = v9 + 1
        v11 = v2 * v10
        return v11
# Inputs to the model
x1 = torch.randn(1, 3, 5, 7)
