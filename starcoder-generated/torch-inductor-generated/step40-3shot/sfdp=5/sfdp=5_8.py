
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 6
        self.seq_len = 7782
        self.dim = 384 // self.heads
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.1, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(2, 6, 2, 384)
key = torch.randn(2, 6, 2, 384)
value = torch.randn(2, 6, 2, 384)
attn_mask = torch.randn(2, 1, 2, 2)
