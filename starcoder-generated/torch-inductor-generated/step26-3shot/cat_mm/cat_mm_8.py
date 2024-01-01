
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t1 = torch.mm(x1, x2)
        self.v1 = torch.mm(x1, x2)
        self.t2 = torch.cat([self.t1, self.t1, self.t1], 1)
    def forward(self, x1, x2):
        v0 = torch.mm(x1, x2)
        v2 = torch.mm(x1, x2)
        return torch.cat([v0, v0, v0, v2, v2], 1)
# Inputs to the model
x1 = torch.randn(2, 1)
x2 = torch.randn(1, 4)
