
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp1, inp2):
        t1 = torch.mm(x1, inp1)
        t2 = torch.mm(inp2, x1)
        t3 = torch.mm(x2, inp2)
        t4 = x1 + inp2
        t5 = t3 + inp2
        return t4
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
inp1 = torch.randn(3, 3)
inp2 = torch.randn(3, 3)
