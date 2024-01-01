
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Linear(100, 100)
        self.query = torch.nn.Linear(100, 100)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, query, key):
        q = self.query(query)
        k = self.key(key)
        a = torch.matmul(q, k.transpose(-2, -1))
        s = a.div(10.0)
        m = s.softmax(dim=-1)
        d = self.dropout(m)
        o = torch.matmul(d, s)
        return o
# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 5, 100)
key = torch.randn(5, 100, 64, 64)
