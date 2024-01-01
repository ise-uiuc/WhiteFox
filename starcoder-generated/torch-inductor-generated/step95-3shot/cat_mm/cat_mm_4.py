
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        t1 = torch.mm(x1, x2)
        t2 = torch.mm(x1, x2)
        t = torch.cat([t1, t2])
        return torch.cat([t] * 20, 1)
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(2, 2)
