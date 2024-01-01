
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(95, 91, 1, stride=1, padding=22)
    def forward(self, x21):
        v1 = self.conv(x21)
        v2 = v1 * 0.5
        v3 = v1 * v1
        v4 = v3 * v1
        v5 = v4 * 0.044715
        v6 = v1 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v2 * v8
        return v9
# Inputs to the model
x21 = torch.randn(1, 95, 22, 93)
