
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = x1 * 0.5
        v2 = x1 * x1
        v3 = x1 * x2
        v4 = x1 + v2
        v5 = x1 * v3
        v6 = torch.tanh(x1)
        v7 = v6 * 0.044715
        v8 = v9 * v6
        v9 = v5 * 0.5
# Inputs to the model
x1 = torch.randn(1, 4, 32, 32)
x2 = torch.randn(1, 4, 32, 32)
