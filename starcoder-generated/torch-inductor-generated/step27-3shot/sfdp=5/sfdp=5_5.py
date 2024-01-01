
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = 0.2
        self.heads = 16
        self.seq_len = 128
        self.dim = 128 // self.heads
    def forward(self, query, key, value, attn_mask=None):
        attn_mask = torch.zeros(query.shape[0], key.shape[1], key.shape[2], key.shape[2])
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, self.dropout, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 16, 128, 128)
key = torch.randn(1, 16, 128, 128)
value = torch.randn(1, 16, 128, 128)
