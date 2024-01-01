
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x1.permute(0, 3, 2, 1)
        v2 = x2.permute(0, 3, 1, 2)
        v3 = torch.matmul(v1, v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 3, 2)
x2 = torch.randn(1, 1, 3, 2)
