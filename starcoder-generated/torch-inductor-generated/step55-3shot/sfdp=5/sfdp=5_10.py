
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 128
        self.seq_len = 768
        self.dim = 896 // self.heads
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.6, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 192, 768, 896)
key = torch.randn(1, 192, 768, 896)
value = torch.randn(1, 192, 768, 896)
attn_mask = torch.randn(1, 1, 768, 768)
