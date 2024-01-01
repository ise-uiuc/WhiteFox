
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inp1 = torch.randn(3, 3)
        self.inp2 = torch.randn(3, 3)
    def forward(self, x1):
        v2 = self.inp1 + self.inp2
        v1 = torch.mm(x1, v2)
        return v1
# Inputs to the model
x1 = torch.randn(3, 3)
