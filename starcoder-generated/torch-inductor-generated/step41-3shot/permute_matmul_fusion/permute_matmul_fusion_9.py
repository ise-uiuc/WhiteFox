
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x1.permute(0, 1, 3, 2)
        v2 = torch.bmm(v1, x2)
        v3 = torch.matmul(v2, v1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 2, 3)
x2 = torch.randn(1, 1, 2, 3)
