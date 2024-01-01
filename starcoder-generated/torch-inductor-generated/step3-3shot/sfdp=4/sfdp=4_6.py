
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, attn_mask=None):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        if attn_mask is not None:
            qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ value
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(2, 16, 64, 64)
key = torch.randn(2, 8, 64, 64)
value = torch.randn(2, 8, 64, 64)
