
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 30, 1, stride=1, padding=0)
    def forward(self, x2):
        v1 = self.conv(x2)
        v4 = v1 * 0.044715
        v5 = v1 + v4
        v8 = v5 * 0.7978845608028654
        v9 = torch.tanh(v8)
        v10 = v9 + 1
        return v10
# Inputs to the model
x2 = torch.randn(1, 1, 15, 15)
