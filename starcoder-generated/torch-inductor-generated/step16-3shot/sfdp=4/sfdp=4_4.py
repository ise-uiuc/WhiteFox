
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.nhead = 6

    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ value
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(10, 12, 64)
key = torch.randn(10, 8, 64)
value = torch.randn(10, 8, 64)
