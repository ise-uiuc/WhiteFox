
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv7 = torch.nn.Conv2d(4, 5, 5, stride=13, padding=0)
    def forward(self, x59):
        v1 = self.conv7(x59)
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
x59 = torch.randn(1, 4, 31, 28)
