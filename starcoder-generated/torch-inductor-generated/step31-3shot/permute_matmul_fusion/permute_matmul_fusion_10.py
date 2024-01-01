
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x2.permute(0, 2, 1)
        v2 = x1.permute(0, 2, 1)
        v3 = torch.matmul(v1, v2)
        v4 = torch.matmul(v2, x2)
        v5 = v2.permute(0, 2, 1)
        w1 = v4 * x1
        return (v3 - torch.bmm(v5, w1))
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
