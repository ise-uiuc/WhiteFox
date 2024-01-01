
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 512
        self.seq_len1 = 128
        self.seq_len2 = 1024
        self.dim = 512 // self.heads
    def forward1(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.6, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(16, 1024, 128, 512)
key = torch.randn(16, 1024, 128, 512)
value = torch.randn(16, 1024, 128, 512)
attn_mask = torch.randn(16, 1, 128, 128)
