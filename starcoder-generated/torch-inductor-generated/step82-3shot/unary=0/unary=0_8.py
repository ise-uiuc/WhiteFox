
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(21, 47, 1, stride=1, padding=0)
    def forward(self, x37):
        v1 = self.conv(x37)
        v2 = v1 * 0.035123
        v3 = v1 + v2
        v4 = v3 * 0.42611
        v5 = v3 + v4
        v6 = v5 * 0.55769
        v7 = torch.tanh(v6)
        v8 = v7 + 1
        v9 = v5 * v8
        return v9
# Inputs to the model
x37 = torch.randn(1, 21, 42, 54)
