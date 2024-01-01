
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 16
        self.dim = 16
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim = -1)
        attn_weight = torch.dropout(attn_weight, 0.5, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(2, 16, 19, 128)
key = torch.randn(2, 16, 19, 128)
value = torch.randn(2, 16, 19, 128)
attn_mask = torch.randn(2, 1, 19, 19)
