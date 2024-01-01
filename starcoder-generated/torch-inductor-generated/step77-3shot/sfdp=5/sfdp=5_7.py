
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 12
        self.seq_len = 48
        self.dim = 256 // self.heads
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.2, True)
        output = attn_weight @ value
        return attn_weight.max(dim=-1)[0].sum()
# Inputs to the model
query = torch.randn(1, 12, 48, 256)
key = torch.randn(1, 12, 48, 256)
value = torch.randn(1, 12, 48, 256)
attn_mask = torch.randn(1, 1, 48, 48)
