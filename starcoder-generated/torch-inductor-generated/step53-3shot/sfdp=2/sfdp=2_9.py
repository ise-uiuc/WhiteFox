
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(256, 128, bias=False)
        self.dropout = torch.nn.Dropout(0.1)
        self.linear2 = torch.nn.Linear(128, 256, bias=False)
 
    def forward(self, x1, x2):
        y1 = self.linear1(x1)
        y2 = self.linear2(x2)
        z1 = torch.matmul(y1, y2.transpose(-2, -1))
        z2 = self.dropout(z1)
        z3 = torch.matmul(z2, y1)
        return z1, z3
 
m = Model()

x1 = torch.randn(1, 256, 10)
x2 = torch.randn(1, 10, 256)
x3, x4 = m(x1, x2)

