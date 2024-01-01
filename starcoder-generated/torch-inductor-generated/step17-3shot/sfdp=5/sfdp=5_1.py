
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = 0.1
        self.heads = 16
        self.seq_len = 6
        self.dim = 512 // self.heads
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, self.dropout, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(64, 16, 6, 512)
key = torch.randn(64, 16, 6, 512)
value = torch.randn(64, 16, 6, 512)
attn_mask = torch.randn(64, 1, 6, 6)
