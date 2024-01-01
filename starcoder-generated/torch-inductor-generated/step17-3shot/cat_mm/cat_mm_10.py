
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.f0 = torch.nn.Linear(5, 6)
    def forward(self, x1, x2):
        t1 = torch.cat([torch.mm(x1, x2),  torch.mm(x1, x2)], 1)
        t2 = self.f0(t1)
        t3 = self.f0(t1)
        return torch.cat([t2, t3], 1)
# Inputs to the model
x1 = torch.randn(3, 5)
x2 = torch.randn(5, 1)
