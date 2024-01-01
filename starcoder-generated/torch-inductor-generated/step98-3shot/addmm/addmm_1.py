
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, x2):
        t1 = torch.mm(x, x2)
        t2 = t1 + x
        return t2
# Inputs to the model
x = torch.randn(3, 3)
x2 = torch.randn(3, 3)
