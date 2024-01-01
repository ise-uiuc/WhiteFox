
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(10, 16, 4, stride=1, padding=1)
        self.conv1 = torch.nn.Conv2d(16, 25, 5, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v1 * v1
        v4 = v3 * v1
        v5 = v4 * 0.044715
        v6 = v1 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v2 * v9
        v11 = self.conv1(v10)
        v12 = 0.010981*v11
        v13 = torch.sigmoid(v11)
        v14 = v11*v13
        v15 = 0.009767*v14
        v16 = v13 + v15
        v17 = torch.min(v16, 1)
        v18 = v17[0]*v13*v16
        return v18
# Inputs to the model
x1 = torch.randn(1, 10, 17, 19)
