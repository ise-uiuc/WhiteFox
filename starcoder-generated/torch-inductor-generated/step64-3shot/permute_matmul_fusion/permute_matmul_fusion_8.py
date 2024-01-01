
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v0 = x1.permute(2, 0, 1)
        v1 = x2.permute(2, 0, 1)
        v2 = torch.matmul(v1, x1)
        return v2.permute(1, 2, 0)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
