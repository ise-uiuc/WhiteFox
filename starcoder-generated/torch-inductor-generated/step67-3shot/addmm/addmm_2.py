
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.v1 = torch.randn(3, 3)
        self.v2 = torch.randn(3, 3)
        self.v2 = torch.randn(3, 3)
    def forward(self, x1, x2, inp=None):
        if inp is not None:
            v2 = self.v2
        else:
            v2 = x2
        v1 = torch.mm(self.v1, self.v1)
        v2 = torch.mm(v1, v2)
        v3 = v1 + v2
        return v3
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3, requires_grad=True)
