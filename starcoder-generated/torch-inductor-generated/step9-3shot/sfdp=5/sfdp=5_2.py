
class Model(torch.nn.Module):
    def __init__(self, n_head, n_query_head):
        super().__init__()
        self.query = torch.nn.Linear(10, n_query_head)
        self.key = torch.nn.Linear(10, n_head)
        self.value = torch.nn.Linear(10, n_head)
        
    def forward(self, query, key, value):
        qk = self.query(query) @ self.key(key).transpose(-1, -2)
        qk = qk / math.sqrt(query.size(-1))
        attn_mask = torch.eye(qk.size(-2), device=qk.device)[None, None, :, :] * -1e10 
        qk = qk + attn_mask
        
        attn_weight = torch.softmax(qk, dim=-2)
        attn_weight = torch.nn.functional.dropout(attn_weight, p=0.5, training=True)
        
        output = attn_weight @ self.value(value).transpose(-1, -2)
        return output

# Initializing the model
m = Model(n_head=2, n_query_head=2, value_dim=5, dropout_p=0.5)

# Inputs to the model
query = torch.randn(1, 2, 10)
key = torch.randn(1, 2, 10)
value = torch.randn(2, 4, 5)
