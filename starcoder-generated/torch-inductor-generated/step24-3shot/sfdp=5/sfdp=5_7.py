
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 256
        self.seq_len = 512
        self.dim = 111
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.1, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 256, 64, 56)
key = torch.randn(1, 256, 64, 56)
value = torch.randn(1, 256, 64, 56)
attn_mask = torch.randn(1, 1, 64, 64)
