
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 32
        self.seq_len = 32
        self.dim = 1024 // self.heads
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1)) # (32, 32, 1024)
        qk = qk + attn_mask.to(query)
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.2, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(2, 32, 32, 1024)
key = torch.randn(2, 32, 32, 1024)
value = torch.randn(2, 32, 32, 1024)
attn_mask = torch.randn(2, 1, 32, 32)
