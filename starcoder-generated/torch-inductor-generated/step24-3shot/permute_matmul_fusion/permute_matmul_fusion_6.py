
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x1.transpose(0, 2)
        v2 = x2.permute(0, 2, 1)
        v3 = torch.matmul(v2, v1)
        v4 = torch.matmul(v1, v3)
        v5 = v4 * v2
        v6 = v5 - v1
        v7 = v4 + v6
        return (v4, v7)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
