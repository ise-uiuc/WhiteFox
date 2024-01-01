
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.nn.functional.relu(x1)
        v2 = torch.nn.functional.relu6(x1)
        v3 = v1 + 1
        v4 = v3 * 6
        v5 = v4 / 6
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
