
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.randn((66, 43, 10, 10))
    def forward(self, x1):
        v1 = torch.abs(x1 - self.weight)
        v2 = torch.sum(v1, axis=(0))
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
x1 = torch.randn(5, 5, 5, 5)
