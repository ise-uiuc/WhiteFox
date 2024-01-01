
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 1534
        self.seq_len = 3495
        self.dim = 199 // self.heads
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.2, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 1534, 3495, 199)
key = torch.randn(1, 1534, 3495, 199)
value = torch.randn(1, 1534, 3495, 199)
attn_mask = torch.randn(1, 1, 3495, 3495)
