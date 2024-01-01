
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 8
        self.seq_len = 1
        self.dim = 6144 // self.heads
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk.transpose(-1, -2).transpose(-3, -4)
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        del qk
        attn_weight = torch.dropout(attn_weight, 2.3897206599917977e-06, True)
        attn_weight = attn_weight.transpose(-1, -2).transpose(-2, -3)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 6144, 1, 64)
key = torch.randn(1, 6144, 1, 64)
value = torch.randn(1, 6144, 1, 64)
attn_mask = torch.randn(1, 1, 1, 1)
