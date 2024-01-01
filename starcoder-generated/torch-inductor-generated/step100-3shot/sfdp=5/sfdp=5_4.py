
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 63
        self.seq_len = 382
        self.dim = 1895 // self.heads
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.7, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 63, 382, 1895)
key = torch.randn(1, 63, 382, 1895)
value = torch.randn(1, 63, 382, 1895)
attn_mask = torch.randn(1, 1, 382, 382)
