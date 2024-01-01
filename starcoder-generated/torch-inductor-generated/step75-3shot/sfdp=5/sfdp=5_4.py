
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 1000000
        self.seq_len = 1600000
        self.dim = 123 // self.heads
    import time

    s = time.time()

    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.99, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 1000000, 1600000, 123)
key = torch.randn(1, 1000000, 1600000, 123)
value = torch.randn(1, 1000000, 1600000, 123)
attn_mask = torch.randn(1, 1, 1600000, 1600000)
