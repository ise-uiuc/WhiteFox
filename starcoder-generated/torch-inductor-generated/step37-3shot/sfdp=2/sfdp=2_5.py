
class Model(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.query = torch.nn.Linear(input_dim, hidden_dim)
        self.key = torch.nn.Linear(input_dim, hidden_dim)
        self.value = torch.nn.Linear(input_dim, hidden_dim)
    
    def forward(self, query, key, value, scale_factor, dropout_p):
        v1 = self.query(query)
        v2 = self.key(key)
        v3 = self.value(value)
        v4 = torch.matmul(v1, v2.transpose(-2, -1))
        v5 = v4.div(scale_factor)
        v6 = torch.nn.functional.dropout(v5, p=dropout_p)
        v7 = torch.matmul(v6, v3)
        return v7

# Initializing the model
m = Model(32, 64)

# Inputs to the model
q = torch.randn(4, 8, 32)
k = torch.randn(4, 16, 32)
v = torch.randn(4, 16, 32)
scale_factor = torch.randn(1)
dropout_p = 0.0
