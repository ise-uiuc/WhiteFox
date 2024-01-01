
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 276
        self.seq_len = 123
        self.dim = 53 // self.heads
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.3, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 276, 123, 53)
key = torch.randn(1, 276, 123, 53)
value = torch.randn(1, 276, 123, 53)
attn_mask = torch.randn(1, 1, 123, 123)
