
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        v1 = x + x
        v2 = torch.relu(v1)
        v3 = v2 + x
        v4 = torch.relu(v3)
        v5 = v4 + x
        return v5
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
