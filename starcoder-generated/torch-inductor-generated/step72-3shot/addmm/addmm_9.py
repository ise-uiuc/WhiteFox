
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inp1 = torch.arange(6.0).reshape((1,3,2))
    def forward(self, x1, x2, x3):
        v1 = x1 + x2
        v2 = v1 + self.inp1
        return v2 + x3
# Inputs to the model
x1 = torch.randn(1,3, 2)
x2 = torch.randn(1,3, 2)
x3 = torch.randn(1,3, 2)
