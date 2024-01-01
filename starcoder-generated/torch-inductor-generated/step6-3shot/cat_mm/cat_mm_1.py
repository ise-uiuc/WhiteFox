
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        t1 = torch.mm(x1, x2)
        t2 = torch.mm(x1, x2)
        t3 = torch.cat([t1, t2], 1)
        return torch.cat([t1, t2, t3], 1)
# Inputs to the model
x1 = torch.randn(3, 2)
x2 = torch.randn(2, 1)
