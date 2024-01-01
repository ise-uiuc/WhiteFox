
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x1[0: 1, 0: 1]
        v2 = x2.permute(0, 2, 1)
        v3 = torch.matmul(v1, v2)
        v4 = v3.permute(0, 2, 1)
        v5 = v3.permute(2, 0, 1)
        return v4
# Inputs to the model
x1 = torch.randn(3, 4, 5, 6)
x2 = torch.randn(3, 4, 5, 6)
