
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3):
        v1 = x1 + x2 + x3 - x3
        v2 = torch.relu(v1)
        v3 = torch.nn.functional.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = 1
x3 = 1
