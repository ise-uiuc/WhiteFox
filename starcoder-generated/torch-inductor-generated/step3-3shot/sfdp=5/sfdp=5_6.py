
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(6, 8)
        self.k_proj = torch.nn.Linear(6, 8)
        self.v_proj = torch.nn.Linear(6, 8)
        self.out_proj = torch.nn.Linear(8, 6)
 
    def forward(self, query, key, value, attn_mask):
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, True)
        output = attn_weight @ v
        return self.out_proj(output)

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(2, 6, 16)
key = torch.randn(2, 16, 6)
value = torch.randn(2, 16, 6)
attn_mask = torch.randn([2, 6, 16]).softmax(-1).gt(0.0)
