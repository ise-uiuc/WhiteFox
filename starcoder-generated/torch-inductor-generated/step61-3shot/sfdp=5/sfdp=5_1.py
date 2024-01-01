
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = 0.5597
        self.heads = 7975
        self.seq_len = 989815
        self.dim = 32193 // self.heads
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, self.dropout, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 7975, 989815, 32193)
key = torch.randn(1, 7975, 989815, 32193)
value = torch.randn(1, 7975, 989815, 32193)
attn_mask = torch.randn(1, 1, 989815, 989815)
