
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t1 = torch.randn(2, 2)
        self.t2 = torch.randn(2, 2)
        self.t3 = torch.randn(2, 2)
    def forward(self, x1, x2):
        v1 = torch.mm(x1, self.t1)
        v2 = torch.mm(x1, self.t2)
        v3 = torch.mm(x1, self.t3)
        v4 = torch.mm(x2, self.t1)
        return torch.cat([v1, v2, v3, v4], 0)
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(2, 2)
