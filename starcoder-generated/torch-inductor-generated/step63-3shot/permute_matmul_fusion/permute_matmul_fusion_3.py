
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v3 = torch.matmul(x1 * x2, x1.permute(0, 2, 1))
        v2 = torch.matmul(v3, x2)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 2).abs()
x2 = torch.randn(1, 2, 2).abs()
