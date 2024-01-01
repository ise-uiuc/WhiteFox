
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 16
        self.seq_len = 100
        self.dim = 123
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.1, True)
        output = attn_weight @ value
        output = output.transpose(0, 2)
        return output
# Inputs to the model
query = torch.randn(16, 100, 123, 16)
key = torch.randn(16, 100, 123, 16)
value = torch.randn(16, 100, 123, 16)
attn_mask = torch.randn(1, 1, 100, 100)
