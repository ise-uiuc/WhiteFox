
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 8
        self.seq_len = 320
        self.dim = 149 // self.heads
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.31, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(5, 5, 41, 41, 55, 21)
key = torch.randn(5, 5, 41, 41, 55, 21)
value = torch.randn(5, 5, 41, 41, 55, 21)
attn_mask = torch.randn(5, 5, 41, 41)
