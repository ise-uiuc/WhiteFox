
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, attn_mask, dropout_p):
        a1 = query @ key.transpose(-2, -1)
        a1 = a1 / math.sqrt(query.size(-1))
        a1 = a1 + attn_mask
        attn_weight = torch.softmax(a1, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, True)
        output = attn_weight @ value
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(8, 3, 20)
key = torch.randn(8, 3, 20)
value = torch.randn(8, 3, 20)
attn_mask = torch.randn(8, 3, 3)
dropout_p = 0.5
