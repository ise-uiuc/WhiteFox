
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(26, 13, 5, stride=2, padding=1, groups=13)
        self.conv_2 = torch.nn.Conv2d(13, 11, 1, stride=1, padding=0, groups=1)
    def forward(self, x5):
        v1 = self.conv_1(x5)
        v2 = v1 * 0.5
        v3 = v1 * v1
        v4 = v3 * v1
        v5 = v4 * 0.044715
        v6 = v1 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v2 * v9
        w1 = self.conv_2(v10)
        w2 = w1 * 0.5
        w3 = w1 * w1
        w4 = w3 * w1
        w5 = w4 * 0.044715
        w6 = w1 + w5
        w7 = w6 * 0.7978845608028654
        w8 = torch.tanh(w7)
        w9 = w8 + 1
        w10 = w2 * w9
        return w10
# Inputs to the model
x5 = torch.randn(1, 26, 112, 112)
