
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(11, 8, 2, stride=1, padding=1)
        self.dropout = torch.nn.Dropout(p=0.3)
    def forward(self, x3):
        v1 = self.conv(x3)
        v2 = self.dropout(v1)
        v3 = v2 * 0.5
        v4 = v2 * v2
        v5 = v4 * v2
        v6 = v5 * 0.044715
        v7 = v2 + v6
        v8 = v7 * 0.7978845608028654
        v9 = torch.tanh(v8)
        v10 = v9 + 1
        v11 = v3 * v10
        return v11
# Inputs to the model
x3 = torch.randn(1, 11, 64, 64)
