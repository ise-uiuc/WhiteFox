
class Model(torch.nn.Module):
    def __init__(self, key_size, query_size, value_size, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.key = torch.nn.Linear(key_size, hidden_dim)
        self.query = torch.nn.Linear(query_size, hidden_dim)
        self.value = torch.nn.Linear(value_size, hidden_dim)
        
    def forward(self, x1, x2):
        k = self.key(x1)
        q = self.query(x2)
        v = self.value(x3)
    
        inv_scale = 1 / math.sqrt(self.hidden_dim)
        scaled_dp = torch.matmul(k, q.T) * inv_scale
        attention_weights = scaled_dp.softmax(dim=-1)
        output = attention_weights.matmul(v)
    
        return output

# Initializing the model
model = Model(key_size=64, query_size=64, value_size=64, hidden_dim=64)

# Inputs to the model
x1 = torch.randn(1, 64)
x2 = torch.randn(1, 64)
x3 = torch.randn(1, 64)

