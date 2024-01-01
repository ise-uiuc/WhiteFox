
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        q = a @ b.T
        sc = q / 10
        sm = torch.nn.functional.softmax(sc, dim=-1)
        dr = torch.nn.functional.dropout(sm, 0.2)
        out = dr @ a
        o = self.out(out)
        return o

# Initializing the model
m = Model()

# Inputs to the model
a = torch.randn(1, 12, 32)
b = torch.randn(12, 32)
