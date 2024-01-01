
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1, stride=1, padding=1)
    def forward(self, x23):
        v1 = self.conv(x23)
        v4 = v1 * 0.044715
        v3 = v1 + v4
        v7 = v3 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v2 = v1 * 0.5
        v10 = v2 * v9
        return v10
# Inputs to the model
x23 = torch.randn(10, 1, 4, 4)
