
class Model(torch.nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
 
        self.w_q = torch.nn.Linear(hidden_dim, hidden_dim)
        self.w_k = torch.nn.Linear(hidden_dim, hidden_dim)
        self.w_v = torch.nn.Linear(hidden_dim, hidden_dim)
 
    def forward(self, queries, keys, values, attn_mask):
        q = self.w_q(queries)
        k = self.w_k(keys)
        v = self.w_v(values)
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = (attn_weight @ v)
        return output

# Initializing the model to have hidden_dim be 512
m = Model(hidden_dim=512)

# Input Tensors to the model
queries = torch.randn(1, 8, 512)
keys = torch.randn(1, 8, 512)
values = torch.randn(1, 8, 512)
attn_mask = torch.randn(1, 8, 1, 1)
output = m(queries, keys, values, attn_mask)

