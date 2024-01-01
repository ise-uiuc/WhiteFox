
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        k = torch.matmul(x1, x2.transpose(-2, -1))
        v = torch.matmul(x1, x2.transpose(-2, -1))
        q = torch.matmul(x1, x2.transpose(-2, -1))
        s = k.mul(0.15)
        t = s.softmax(-1)
        f = torch.nn.functional.dropout(t, 0.4)
        o = torch.matmul(f, x2)
        return o

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4, 64)
x2 = torch.randn(1, 8, 64)
