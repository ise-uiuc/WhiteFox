
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 4, stride=1, padding=1)

        self.conv2 = torch.nn.Conv2d(4, 2, 5, stride=1, padding=0)
    def forward(self, x5):
        v1 = self.conv(x5)
        v2 = v1 * 0.5
        v3 = v1 * v1
        v4 = v3 * v1
        v5 = v4 * 0.044715
        v6 = v1 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v2 * v9

        v14 = self.conv2(v10)
        v17 = v14 * 0.060631
        v19 = v17 + 1
        return v19
# Inputs to the model
x5 = torch.randn(1, 3, 64, 64)
