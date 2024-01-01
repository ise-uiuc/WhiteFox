
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x1.permute(0, 2, 1)
        v3 = x2.sum(dim=-1)[..., None]
        v2 = v1.permute(0, 2, 1)
        return torch.matmul(x2 * v3, v1.permute(0, 2, 1))
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
