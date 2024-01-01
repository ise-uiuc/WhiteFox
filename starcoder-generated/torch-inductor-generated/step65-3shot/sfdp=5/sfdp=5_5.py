
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 1024
        self.seq_len = 64
        self.dim = 1
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask[0][0]
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 1024, 64, 1)
key = torch.randn(1, 1024, 64, 1)
value = torch.randn(1, 1024, 64, 1)
attn_mask = torch.randn(1, 1, 64, 64)
