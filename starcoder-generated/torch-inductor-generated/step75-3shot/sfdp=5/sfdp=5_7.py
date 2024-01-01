
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 1536
        self.seq_len = 256
        self.dim = 16384 // self.heads
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.4, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 1536, 256, 16384)
key = torch.randn(1, 1536, 256, 16384)
value = torch.randn(1, 1536, 256, 16384)
attn_mask = torch.randn(1, 1, 256, 256)
