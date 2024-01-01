
class Model(torch.nn.Module):
    def __init__(self, attn_mask):
        super().__init__()
        self.attn_mask = attn_mask
 
    def forward(self, query, key, value):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + self.attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ value
        return output

# Initializing the model
attn_mask = torch.arange(8).view(8, 1, 1, 1).expand(8, -1, 64, 64)
m = Model(attn_mask)

# Inputs to the model
query = torch.randn(8, 2, 64, 64)
key = torch.randn(8, 2, 64, 64)
value = torch.randn(8, 2, 64, 64)
