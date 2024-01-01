
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x):
        q = x.matmul(y)
        k = z.matmul(w).transpose(-2, -1)
        v = u.matmul(l)
        scores = q.matmul(k).div(0.5)
        out = dropout(scores)
        out = out.matmul(v)
        return out, values

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(5, 4, 4)
__output__, __values__ = m(x)

