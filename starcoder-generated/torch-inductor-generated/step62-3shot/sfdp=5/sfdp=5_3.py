
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 64
        self.seq_len = 210
        self.dim = 13
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.4, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 64, 210, 13)
key = torch.randn(1, 64, 210, 13)
value = torch.randn(1, 64, 210, 13)
attn_mask = torch.randn(1, 1, 210, 210)
