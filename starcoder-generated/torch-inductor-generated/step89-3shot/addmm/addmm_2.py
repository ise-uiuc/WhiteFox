
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.ones(32, 32, requires_grad=True)
    def forward(self, x1, x2, inp):
        v1 = x1 @ x2
        v2 = torch.mm(self.op_mm_w(), v1) + inp
        return v2
    def op_mm_w(self):
        return torch.mm(self.w, self.w.t())
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
inp = torch.randn(3, 3)
