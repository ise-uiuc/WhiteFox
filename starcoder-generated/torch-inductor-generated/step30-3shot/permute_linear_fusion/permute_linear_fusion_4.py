
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v3 = torch.matmul(v1, v1)
        v4 = v3.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v4, v4, v4)
        return v2
# Inputs to the model
x1 = torch.randn(3, 18, 10)
