
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(16, 8)
        self.k = torch.nn.Linear(16, 9)
        self.v = torch.nn.Linear(16, 6)
 
    def forward(self, query, key, value):
        q = torch.relu(self.q(query))
        k = torch.relu(self.k(key))
        v = torch.relu(self.v(value))

        d1 = torch.matmul(q, k.transpose(2, 1))
        d2 = d1 / math.sqrt(d1.shape[2])
        d3 = torch.nn.functional.dropout(d2, p=0.98, training=False)
        d4 = torch.matmul(d3, v)

        return d4

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 4, 16)
key = torch.randn(1, 5, 16)
value = torch.randn(1, 4, 16)
