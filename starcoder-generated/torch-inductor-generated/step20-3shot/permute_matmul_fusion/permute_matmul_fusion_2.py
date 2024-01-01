
class Model1(torch.nn.Module):
    def __init__(self, A):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x1.permute(0, 2, 1)
        v2 = x2.permute(0, 2, 1)
        v3 = torch.bmm(v1, v2)
        return v3.permute(0, 2, 1)
# Inputs to the model
A = torch.randn(1, 2, 2)
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
