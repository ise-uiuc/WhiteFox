
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = 0.1
        self.heads = 32
        self.seq_len = 384
        self.dim = 160 // self.heads
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.nn.Softmax(dim=-1)(qk)
        attn_weight = torch.nn.Dropout(self.dropout)(attn_weight)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 32, 384, 160)
key = torch.randn(1, 32, 384, 160)
value = torch.randn(1, 32, 384, 160)
attn_mask = torch.randn(1, 1, 384, 384)
