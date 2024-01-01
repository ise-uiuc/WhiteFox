
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = 0.1
        self.heads = 32
        self.seq_len = 256
        self.dim = 128 // self.heads
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, self.dropout, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 32, 256, 128)
key = torch.randn(1, 32, 256, 128)
value = torch.randn(1, 32, 256, 128)
attn_mask = torch.randn(1, 1, 256, 256)
