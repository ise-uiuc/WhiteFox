
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.mm(x1, x1)
        v2 = torch.mm(x1, v1)
        v3 = x1 + v1
        v4 = v2 + v3
        return v3 + v4
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
