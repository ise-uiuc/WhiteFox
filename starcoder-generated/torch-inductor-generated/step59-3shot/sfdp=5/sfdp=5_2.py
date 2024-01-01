
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 128
        self.seq_len = 768
        self.dim = 768
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 128, 768, 768)
key = torch.randn(1, 128, 768, 768)
value = torch.randn(1, 128, 768, 768)
attn_mask = torch.randn(1, 1, 32, 32)
