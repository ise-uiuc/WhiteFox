
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x, y, z):
        w = torch.matmul(x, y)
        v = w.div(100.0)
        t = v.softmax(dim=-1)
        u = torch.nn.functional.dropout(t, p=0.3)
        h = torch.matmul(u, z)
        return h

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(15, 20, 20)
y = torch.randn(15, 20, 30)
z = torch.randn(15, 30, 10)
