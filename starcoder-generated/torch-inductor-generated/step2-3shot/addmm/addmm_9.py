
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = torch.mm(x1, x2)
        t1 = v1 + inp
        t2 = torch.sigmoid(t1)
        t3 = t2 * x1
        t4 = t3[0:3, 2]
        t5 = t4.unsqueeze(1)
        t6 = t2[:4, :]
        return t5, t6
# Inputs to the model
x1 = torch.randn(3, 4)
x2 = torch.randn(4, 8)
inp = torch.randn(3, 3)
