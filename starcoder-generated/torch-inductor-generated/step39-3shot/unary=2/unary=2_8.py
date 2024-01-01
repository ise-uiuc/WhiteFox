
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = x1
        v2 = x1 * x1
        v3 = x1 * v1
        v4 = v2 + v4
        v5 = x1 * v4
        v6 = 0.47320427255675897 # v6 is set to 0.47320427255675897
        v7 = v5 * v6
        return v7
# Inputs to the model
x1 = torch.randn(16, 16)
