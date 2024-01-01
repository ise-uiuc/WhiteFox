
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t1 = torch.mm(x, x)
        t2 = torch.cat([t1, t1, t1], 1)
        return torch.mm(t2, t2)
# Inputs to the model
x = torch.randn(2, 1)
