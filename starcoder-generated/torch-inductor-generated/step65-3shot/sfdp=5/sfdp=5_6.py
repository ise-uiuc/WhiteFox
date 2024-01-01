
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 8
        self.seq_len = 111
        self.dim = 92 // self.heads
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.015873015873015872, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 8, 111, 92)
key = torch.randn(1, 8, 111, 92)
value = torch.randn(1, 8, 111, 92)
attn_mask = torch.randn(1, 1, 111, 111)
