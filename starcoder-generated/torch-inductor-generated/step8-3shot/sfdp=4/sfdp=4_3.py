
class Model(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
        self.query_layer = torch.nn.Linear(p[0], p[1], bias=False)
        self.key_layer = torch.nn.Linear(p[0], p[1], bias=False)
        self.value_layer = torch.nn.Linear(p[0], p[1], bias=False)
 
    def forward(self, q, k, v, mask):
        q = self.query_layer(q)
        k = self.key_layer(k)
        v = self.value_layer(v)
        q = q / math.sqrt(q.size(-1))
        qk = torch.matmul(q, k.transpose(-2, -1))
        qk += mask
        weights = torch.nn.functional.softmax(qk, dim=-1)
        output = torch.matmul(weights, v)
        return output

# Initializing the model
p = [12, 12]       # Size of query, key, and value
m = Model(p)

# Inputs to the model
q = torch.randn(1, 3, p[0])
k = torch.randn(2, 3, p[0])
v = torch.randn(2, 3, p[0])
mask = torch.full((2, 3), -float('Inf'))
