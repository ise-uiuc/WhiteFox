
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(56, 8)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.linear(c1)
        v4 = self.linear(c1)
        v5 = m(c2)
        v6 = v2 * v3
        v7 = v3 * v4
        v8 = v1 * v5
        return v6 + v7 + v8