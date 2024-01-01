
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, key, value, query, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, True)
        output = attn_weight @ value
        return output

# Initializing the model
m = Model()

# Inputs to the model
key = torch.randn(4, 1, 2, 3)
value = torch.randn(4, 5, 2, 3)
query = torch.randn(4, 6, 2, 3)
attn_mask = torch.randn(4, 6, 6, 1)
output = m(key, value, query, attn_mask)

