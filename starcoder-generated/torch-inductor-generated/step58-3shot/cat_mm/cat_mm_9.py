
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        t1 = torch.mm(x1, x2)
        t2 = torch.mm(x1, x2)
        t3 = torch.mm(x2, t1)
        t4 = torch.mm(t2, t1)
        t5 = torch.mm(t1, t1)
        t6 = torch.mm(x1, x1)
        return torch.cat([t1, t2, t3, t4, t5, t6], 1)
# Inputs to the model
x1 = torch.randn(4, 4)
x2 = torch.randn(8, 4)
