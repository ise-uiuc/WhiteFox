
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.v1 = torch.nn.Parameter(torch.randn(2, 3))
        self.v2 = torch.nn.Parameter(torch.randn(3, 4))
        self.v3 = torch.nn.Parameter(torch.randn(5, 3))
        self.v4 = torch.nn.Parameter(torch.randn(13, 11))
        self.v5 = torch.nn.Parameter(torch.randn(1, 13))
    def forward(self, x1):
        v6 = torch.mm(self.v1, self.v2)
        v7 = torch.mm(self.v3, self.v4)
        v8 = torch.mm(v7, self.v1) # Note, the middle two of the two operands of torch.mm are the same tensor object
        v9 = torch.mm(self.v3, self.v5)
        v10 = torch.mm(v9, x1)
        return torch.cat([v10, v6, v8, v9, v9, v8, v6, v9, v8, v6, v9], 0)
# Inputs to the model
x1 = torch.randn(11, 1)
