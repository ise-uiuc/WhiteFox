
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, p1):
        v1 = torch.mm(x1, x2)
        v2 = v1 + x1
        return v2
# Inputs to the model
x1 = torch.randn(1564, 3676)
x2 = torch.randn(3676, 32)
