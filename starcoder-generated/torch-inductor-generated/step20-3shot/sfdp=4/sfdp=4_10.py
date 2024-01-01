
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ value
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 8, 64)
key = torch.randn(1, 8, 64)
value = torch.randn(1, 8, 64)
attn_mask = torch.randn(8, 8)
