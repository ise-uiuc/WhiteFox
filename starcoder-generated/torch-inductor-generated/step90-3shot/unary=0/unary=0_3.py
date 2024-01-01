
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv30 = torch.nn.Conv2d(21, 93, 43, stride=10, padding=17)
    def forward(self, x92):
        v1 = self.conv30(x92)
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
# Inputs to the model
x92 = torch.randn(1, 21, 405, 32)
