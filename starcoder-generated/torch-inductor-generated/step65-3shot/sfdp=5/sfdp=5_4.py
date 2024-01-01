
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_len = 4256
        self.dim = 834
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.07597718688141708, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 73, 4256, 834)
key = torch.randn(1, 73, 4256, 834)
value = torch.randn(1, 73, 4256, 834)
attn_mask = torch.randn(1, 1, 4256, 4256)
