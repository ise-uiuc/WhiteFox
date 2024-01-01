
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 128
        self.query_len = 768
        self.key_len = 4
        self.dim = 64 // self.heads
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.8, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 1, 768, 128)
key = torch.randn(1, 4, 768, 32)
value = torch.randn(1, 4, 768, 32)
attn_mask = torch.randn(1, 1, 768, 768)
