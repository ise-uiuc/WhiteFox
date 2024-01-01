
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = x1.transpose(1, 2)
        v2 = torch.tensor([1, 2, 3], dtype=torch.float32).view(1, 1, 3)
        v3 = v1 ** 2 - v2 + v1.transpose(1, 2)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
