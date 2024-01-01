
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x1.permute(0, 2, 1)
        v2 = x1 * x1
        v3 = x1 * x1
        v4 = x2
        v5 = x1 * x1
        v6 = x1 * x1 * x1 * x1
        v7 = x1 * x1 * x1 * x1
        v8 = x2 * x2
        return (v1, v2, v3, v4, v5, v6, v7, v8)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
