
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        v1 = x.expand(1, 16, 64, 64)
        v2 = torch.relu(x)
        v3 = v2 + v1
        return v3
# Inputs to the model:
x = torch.randn(1, 16, 64, 64)
