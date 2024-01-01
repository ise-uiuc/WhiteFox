
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3 = torch.nn.Conv2d(72, 75, 1, stride=2, padding=2)
        self.batch_norm1 = torch.nn.BatchNorm2d(75)
        self.conv4 = torch.nn.Conv2d(75, 78, 1, stride=1, padding=0)
    def forward(self, x58):
        v1 = self.conv3(x58)
        v2 = v1 * 0.5
        v3 = v1 * v1
        v4 = v3 * v1
        v5 = v4 * 0.044715
        v6 = v1 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v2 * v9
        a1 = self.batch_norm1(v10)
        v11 = self.conv4(a1)
        v12 = v11 * 0.5
        return v12
# Inputs to the model
x58 = torch.randn(1, 72, 46, 70)
