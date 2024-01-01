
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v2 = torch.mm(x1, x2)
        v3 = torch.mm(v2, x1) + x1
        v4 = torch.mm(v3, x2)
        return v4
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3)
