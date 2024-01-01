
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t1 = torch.randn(3, 3, requires_grad=True)
    def forward(self, x1, inp):
        v1 = torch.mm(x1, x1) + self.t1
        return v1 * inp
# Inputs to the model
x1 = torch.randn(1, 3)
inp = torch.randn(3, 3)
