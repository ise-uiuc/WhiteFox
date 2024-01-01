
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q, k, v, inf):
        k1 = torch.matmul(q, k.transpose(-2, -1))
        k2 = k1.div(inf)
        k3 = k2.softmax(dim=-1)
        k4 = torch.nn.functional.dropout(k3, p=0.1)
        k5 = torch.matmul(k4, v)
        return k5

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(2, 8, 20)
k = torch.randn(2, 4, 20)
v = torch.randn(2, 4, 24)
inf = 10000
