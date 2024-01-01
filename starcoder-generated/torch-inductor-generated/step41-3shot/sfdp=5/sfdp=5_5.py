
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 137
        self.seq_len = 2001
        self.dim = 209 // self.heads
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.5, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 137, 2001, 209)
key = torch.randn(1, 137, 2001, 209)
value = torch.randn(1, 137, 2001, 209)
attn_mask = torch.randn(1, 1, 2001, 2001)
