
class Model(torch.nn.Module):
    def __init__(self, n_head=4, n_query_dim=32, n_key_dim=32, n_value_dim=32, n_vocab_size=50):
        super().__init__()
        self.scale = math.sqrt(n_query_dim)
 
        self.query = torch.nn.Linear(n_query_dim, n_key_dim)
        self.key = torch.nn.Linear(n_key_dim, n_key_dim)
        self.value = torch.nn.Linear(n_value_dim, n_value_dim)
 
    def forward(self, q, k, v, attn_mask=None):
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)
 
        attn_weight = (q @ k.transpose(-2, -1)) / self.scale
        if attn_mask is not None:
            attn_weight = attn_weight + attn_mask
 
        attn_prob = torch.nn.functional.softmax(attn_weight, dim=-1)
        output = (attn_prob @ v)
 
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(3, 4, 32)
k = torch.randn(3, 7, 32)
v = torch.randn(3, 7, 32)
attn_mask = torch.tril(torch.ones(3, 4, 7)) == 0
