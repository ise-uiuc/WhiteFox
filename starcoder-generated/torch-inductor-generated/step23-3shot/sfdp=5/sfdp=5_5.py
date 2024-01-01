
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 8192
        self.seq_len = 102
        self.dim = 64 // self.heads
        self.dropout = 0.1
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, self.dropout, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 16384, 102, 128)
key = torch.randn(1, 16384, 102, 128)
value = torch.randn(1, 16384, 102, 128)
attn_mask = torch.randn(1, 1, 102, 102)
