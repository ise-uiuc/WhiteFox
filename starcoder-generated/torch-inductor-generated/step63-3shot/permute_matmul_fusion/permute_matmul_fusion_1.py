
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v2 = torch.bmm(x1, x2)
        v1 = torch.bmm(x2.transpose(1, 2), x1.transpose(1, 2))
        v0 = torch.matmul(v1, v2)
        return v0
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
