
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, x4):
        xx1 = torch.mm(x1, x2)
        xx2 = torch.mm(x3, x4)
        xx3 = xx1 + xx2
        return xx3
# Inputs to the model
x1 = torch.randn(1, 65)
x2 = torch.randn(65, 5)
x3 = torch.randn(1, 65)
x4 = torch.randn(65, 5)
