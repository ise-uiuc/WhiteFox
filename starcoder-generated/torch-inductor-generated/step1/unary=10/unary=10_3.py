
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 5)
 
    def forward(self, x):
        v2 = self.linear(x)
        v3 = v2 + 3
        v4 = v3
        m = 6
        n = 0.0
        o = 6
        p = 0.0
        v5 = v4 < o if n < torch.finfo(v4.dtype).max else m
        v6 = v5
        v7 = v6 * o if p < torch.finfo(v6.dtype).max else m
        v8 = v7
        v9 = v4 / v7
        return v9

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 4)
