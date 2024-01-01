
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x1.permute(0, 2, 1)
        v2 = x2.permute(0, 2, 1)
        v3 = torch.matmul(v1, x2)
        v4 = x2.permute(0, 2, 1)
        v5 = torch.matmul(v3, v1)
        return (v3, v4, v5)
# Inputs to the model
x1 = torch.randn(4, 2, 2)  # Permute the last 2 dimensions of this tensor
x2 = torch.randn(4, 2, 5)
