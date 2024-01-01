
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, C):
        v1 = torch.mm(C, x1)
        return v1 + x2
# Inputs to the model
x1 = torch.randn(8)
x2 = torch.randn(8)
C = torch.randn(8, 8, requires_grad=True)
