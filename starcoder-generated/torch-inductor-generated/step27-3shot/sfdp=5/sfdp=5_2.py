
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = 0.4
        self.heads = 1
        self.seq_len = 128
        self.dim = 256 // self.heads
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, self.dropout, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 1, 128, 256)
key = torch.randn(1, 1, 128, 256)
value = torch.randn(1, 1, 128, 256)
attn_mask = torch.randn(1, 1, 128, 128)
