
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.heads = 64
        self.seq_len = 32
        self.dim = 8192 // self.heads
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1)
        qk = qk / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.0, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 64, 32, 8192)
key = torch.randn(1, 64, 32, 8192)
value = torch.randn(1, 64, 32, 8192)
attn_mask = torch.randn(1, 1, 32, 32)
