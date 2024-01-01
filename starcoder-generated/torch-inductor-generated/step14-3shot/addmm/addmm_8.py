
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(v1, x2)
        v3 = v2 + x3
        return v3
# Inputs to the model
x1 = torch.randn(20, 4)
x2 = torch.randn(4, 20)
x3 = torch.randn(1, 1)
