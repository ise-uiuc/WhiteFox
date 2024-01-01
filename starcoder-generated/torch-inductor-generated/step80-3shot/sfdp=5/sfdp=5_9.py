
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 4
        self.seq_len = 32
        self.dim = 4
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(torch.mul(qk, 10000000.), dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.1, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 2, 32, 4)
key = torch.randn(1, 2, 32, 4)
value = torch.randn(1, 2, 32, 4)
attn_mask = torch.randn(1, 1, 32, 32)
