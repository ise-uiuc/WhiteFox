
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 16)
        self.other = torch.nn.Parameter(torch.rand([16]))
 
    def forward(self, x1, other1=None):
        if other1 is not None:
            v1 = self.linear(x1)
            v2 = v1 + other1
            v3 = torch.relu(v2)
            return v3
        else:
            v1 = self.linear(x1)
            v2 = v1 + self.other
            v3 = torch.relu(v2)
            return v3

m = Model()
x1 = torch.randn(1, 3)
