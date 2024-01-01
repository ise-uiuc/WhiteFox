
class Model(torch.nn.Module):
    def __init__(self, query, key, value):
        super().__init__()
        w = torch.randn(query.size(0), query.size(1))
        b = torch.randn(query.size(0))
        self.query = query
        self.key = key
        self.value = value
        self.softmax = torch.nn.Softmax(dim=-1)
        self.weights = torch.nn.Parameter(w)
        self.bias = torch.nn.Parameter(b)
 
    def forward(self, q1):
        v1 = torch.matmul(q1, self.key.transpose(-2, -1))
        v2 = v1.div(0.72)
        v3 = self.softmax(v2)
        v4 = nn.functional.dropout(v3, p=0.3)
        o = torch.matmul(v4, self.value)
        o = o + self.bias
        o = torch.matmul(o, self.weights)
        return o

# Initializing the model
query = torch.randn(10, 20, 64)
key = torch.randn(6, 20, 100)
value = torch.randn(6, 20, 100)
m = Model(query, key, value)

# Inputs to the model
x1 = torch.randn(10, 20, 64)
