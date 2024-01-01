
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 1
        self.seq_len = 300
        self.dim = 300 // self.heads
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.2, False)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 1, 300, 300)
key = torch.randn(1, 1, 300, 300)
value = torch.randn(1, 1, 300, 300)
attn_mask = torch.randn(1, 1, 300, 300)
