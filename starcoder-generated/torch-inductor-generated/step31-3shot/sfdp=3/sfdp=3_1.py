
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.1)
 
    def forward(self, q, k, v):
        s = torch.matmul(q, k).mul(0.1)
        a = s.softmax(dim=-1)
        d = self.dropout(a)
        r = torch.matmul(d, v)
        return r

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 120, 1024)
k = torch.randn(1, 120, 1024)
v = torch.randn(1, 120, 512)
