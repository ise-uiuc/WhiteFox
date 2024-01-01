
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inp1 = torch.randn(3, 3)
        self.inp2 = torch.randn(3, 3, requires_grad=True)
    def forward(self, x1, x2):
        x2 = x2 + self.inp1
        x1 = x1 + self.inp2
        v1 = x1 + x2
        return v1
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3, requires_grad=True)
