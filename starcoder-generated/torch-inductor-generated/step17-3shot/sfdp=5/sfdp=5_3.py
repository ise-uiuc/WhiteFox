
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = 0.0
        self.heads = 8
        self.seq_len = 512
        self.dim = 16 // self.heads
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.dropout(qk, 0.0, True)
        attn_weight = torch.softmax(attn_weight, dim=-1)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 8, 1024, 16)
key = torch.randn(1, 8, 1024, 16)
value = torch.randn(1, 8, 1024, 16)
attn_mask = torch.randn(1, 1, 1024, 1024)
