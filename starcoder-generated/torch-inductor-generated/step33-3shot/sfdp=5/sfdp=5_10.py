
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 64
        self.seq_len = 1024
        self.dim = 1024 // self.heads
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(16, 64, 8192, 1024) # batch, head, sequence len, dim
key = torch.randn(16, 64, 8192, 1024) # batch, head, sequence len, dim
value = torch.randn(16, 64, 8192, 1024) # batch, head, sequence len, dim
attn_mask = torch.randn(16, 1, 8192, 8192) # batch, 1, sequence len, sequence len
