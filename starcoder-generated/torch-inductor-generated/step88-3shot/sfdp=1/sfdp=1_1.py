
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(query_dim, value_dim)
        self.linear2 = torch.nn.Linear(value_dim, value_dim)
        self.linear3 = torch.nn.Linear(value_dim, value_dim)
 
    def forward(self, x1, x2, x3):
        v1 = self.linear1(x1)
        v2 = self.linear2(x2)
        v3 = self.linear3(x3)
        v4 = v1.transpose(-2, -1) * v2
        v5 = v3.transpose(-2, -1) * v2
        v6 = v3.transpose(-2, -1) * v4
        v7 = v6.softmax(dim=-1)
        v8 = torch.nn.functional.dropout(v7, p=0.2)
        return v8.matmul(v3)

# Initializing the model
m = Model()

# Inputs to the model
query_dim = 64
value_dim = 64
x1 = torch.randn(1, query_dim)
x2 = torch.randn(12, value_dim)
x3 = torch.randn(12, value_dim)
