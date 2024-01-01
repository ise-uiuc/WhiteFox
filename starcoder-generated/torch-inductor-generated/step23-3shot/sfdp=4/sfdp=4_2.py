
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weights = torch.softmax(qk, dim=-1)
        output = (attn_weights @ value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.rand(4, 1, 6)
key = torch.rand(4, 6, 2)
value = torch.rand(4, 6, 3)
attn_mask = torch.triu(torch.ones(12, 12), 1).bool()
