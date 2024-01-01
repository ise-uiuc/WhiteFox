
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = x1 + 3
        v2 = torch.nn.functional.relu6(v1)
        v3 = v2 * v2
        v4 = v3 / 6
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
