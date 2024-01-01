
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x1[0].permute(1, 0)
        v2 = x2[0].permute(1, 0)
        v3 = torch.matmul(v1, v2)[0][0]
        return v3
# Inputs to the model
x1 = torch.randn(4, 1, 2)
x2 = torch.randn(4, 1, 2)
