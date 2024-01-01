
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 64
        self.seq_len = 196
        self.dim = 128 // self.heads
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.25, True)
        output = attn_weight @ value
        return output
# Inputs to the model. This input data has self.seq_len and self.heads * seld.dim = 77 and 1152 dimensions, respectively.
query = torch.randn(1, 8, 196, 128)
key = torch.randn(1, 8, 196, 128)
value = torch.randn(1, 8, 196, 128)
attn_mask = torch.randn(1, 1, 196, 196)
