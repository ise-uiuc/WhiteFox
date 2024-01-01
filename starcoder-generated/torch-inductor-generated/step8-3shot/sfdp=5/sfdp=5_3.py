
class Model(torch.nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.query_project = torch.nn.Linear(hidden_dim, hidden_dim)
        self.key_project = torch.nn.Linear(hidden_dim, hidden_dim)
        self.value_project = torch.nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x1, x2, dropout_p):
        q = self.query_project(x1)
        k = self.key_project(x2)
        v = self.value_project(x2)
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, True)
        output = attn_weight @ v
        return output

# Initializing the model
hidden_dim = 64
dropout_p = 0.1
m = Model(hidden_dim)

# Inputs to the model
x1 = torch.randn(1, 4, hidden_dim)
x2 = torch.randn(1, 100, hidden_dim)
attn_mask = torch.triu(torch.ones(1, 4, 100, 100) * float('-inf') / 99, 1)
