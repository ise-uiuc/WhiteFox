
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 4
        self.seq_len = 300
        self.seq_len_2 = 6
        self.dim = 50
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.1, True)
        output = attn_weight @ value
        return output

# Inputs to the model
query = torch.randn(1, 4, 1200, 50)
key = torch.randn(1, 4, 1200, 50)
value = torch.randn(1, 4, 1200, 50)
attn_mask = torch.randn(1, 1, 1200, 900)
