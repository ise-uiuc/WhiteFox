
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = 0.7854209996613659
        self.heads = 64
        self.seq_len = 16384
        self.dim = 64 // self.heads
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, self.dropout, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 128, 16384, 64)
key = torch.randn(1, 128, 16384, 64)
value = torch.randn(1, 128, 16384, 64)
attn_mask = torch.randn(1, 1, 16384, 16384)
