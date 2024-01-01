
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 1
        self.seq_len = 9430
        self.dim = 63 // self.heads
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.7465271926016464, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 3320, 9430, 63)
key = torch.randn(1, 3320, 9430, 63)
value = torch.randn(1, 3320, 9430, 63)
attn_mask = torch.randn(1, 1, 9430, 9430)
