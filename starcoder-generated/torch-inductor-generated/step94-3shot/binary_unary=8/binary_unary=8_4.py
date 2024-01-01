
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.zeros((1, 1, 64, 64))
        v2 = v1 + x1
        v5 = torch.relu(v2)
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
