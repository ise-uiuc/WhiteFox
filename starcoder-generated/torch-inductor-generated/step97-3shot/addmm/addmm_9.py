
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mm(x2, self.inp2)
        return torch.add(x1, v1)
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
self.inp2 = torch.randn(3, 3, requires_grad=True)
