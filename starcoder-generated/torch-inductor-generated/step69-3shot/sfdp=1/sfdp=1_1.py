
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q1, k1, v1, i, o, p=0.15):
        q = torch.matmul(q1, k1.transpose(-2, -1))
        s = q.div(i)
        s = s.softmax(dim=-1)
        d = torch.nn.functional.dropout(s, p=p)
        o = d.matmul(v1)
        return o

# Initializing the model
m = Model()

# Inputs to the model
q1 = torch.randn(1, 3, 50)
k1 = torch.randn(1, 3, 50)
v1 = torch.randn(1, 3, 50)
