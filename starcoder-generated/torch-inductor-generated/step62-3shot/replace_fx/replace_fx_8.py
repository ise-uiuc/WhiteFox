
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.rand_like(x1)
        v2 = torch.rand_like(x1)
        v3 = torch.randn(3)
        v3.repeat(3)
        v4 = torch.nn.functional.dropout(v1 + v2 + v3)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
