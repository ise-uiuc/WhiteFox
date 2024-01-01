
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 4096
        self.seq_len = 3072
        self.dim = 4 // self.heads
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.5088136429351068, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 16, 3072, 4)
key = torch.randn(1, 16, 3072, 4)
value = torch.randn(1, 16, 3072, 4)
attn_mask = torch.randn(1, 1, 3072, 3072)
