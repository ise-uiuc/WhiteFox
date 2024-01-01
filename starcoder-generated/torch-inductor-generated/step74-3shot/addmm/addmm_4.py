
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = torch.randn(3, 3)
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        result = torch.add(torch.mm(v1, self.inp), x2)
        if result.norm() > 1000:
            self.inp = x2 + self.inp + v1
        return result
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
