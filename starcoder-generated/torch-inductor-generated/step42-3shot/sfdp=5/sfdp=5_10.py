
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 8192
        self.seq_len = 384
        self.dim = 1536 // self.heads
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.05, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(2, 8192, 384, 1536)
key = torch.randn(2, 8192, 384, 1536)
value = torch.randn(2, 8192, 384, 1536)
attn_mask = torch.randn(2, 1, 384, 384)
