
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.matrix = torch.randn(3, 3)
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x2, x1)
        return v1 + v2
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
