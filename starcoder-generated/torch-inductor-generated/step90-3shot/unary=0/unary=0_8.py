
x54653 = torch.rand(1, 22, 16, 16)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv6751 = torch.nn.Conv2d(22, 13, 5, stride=23, padding=22)
    def forward(self, x54653):
        v1 = self.conv6751(x54653)
        v2 = v1 * 0.5
        v3 = v1 * v1
        v4 = v3 * v1
        v5 = v4 * 0.044715
        v6 = v1 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v2 * v9
        return v10
