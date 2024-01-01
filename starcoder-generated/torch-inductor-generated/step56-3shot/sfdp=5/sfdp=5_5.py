
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 128
        self.seq_len = 53
        self.dim = 1969 // self.heads
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.0, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 128, 53, 1969)
key = torch.randn(1, 128, 53, 1969)
value = torch.randn(1, 128, 53, 1969)
attn_mask = torch.randn(1, 1, 53, 53)
