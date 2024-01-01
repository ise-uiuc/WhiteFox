
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1x1 = torch.nn.Conv2d(3, 2, 15, stride=5, padding=2)
        self.conv1 = torch.nn.Conv2d(3, 17, 5, stride=2, padding=3)
    def forward(self, x82):
        v1 = self.conv1x1(x82)
        v2 = v1 + v1
        v3 = self.conv1(x82)
        v4 = v3 * 0.5
        v5 = v3 * v3
        v6 = v5 * v3
        v7 = v6 * 0.044715
        v8 = v3 + v7
        v9 = v8 * 0.7978845608028654
        v10 = torch.tanh(v9)
        v11 = v10 + 1
        v12 = v4 * v11
        return v12
# Inputs to the model
x82 = torch.randn(1, 3, 151, 222)
