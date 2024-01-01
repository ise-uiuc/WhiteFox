
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 56
        self.seq_len = 6
        self.dim = 233
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.5, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 56, 6, 233)
key = torch.randn(1, 56, 6, 233)
value = torch.randn(1, 56, 6, 233)
attn_mask = torch.randn(1, 1, 6, 6)
