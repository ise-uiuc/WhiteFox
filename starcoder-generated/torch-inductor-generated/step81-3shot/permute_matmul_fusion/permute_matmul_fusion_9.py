
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.einsum("abef->aebf", x2)
        v2 = torch.matmul(v1, x1)
        return v2.relu()
# Inputs to the model
x1 = torch.randn(1, 2, 2, 2)
x2 = torch.randn(1, 5, 2, 2)
