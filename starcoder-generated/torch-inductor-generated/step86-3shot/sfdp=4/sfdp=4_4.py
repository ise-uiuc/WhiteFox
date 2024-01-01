
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ value
        return output
# Inputs to the model
query8 = torch.randn(1, 16, 200, 352)
key23 = torch.randn(1, 16, 200, 352)
value7 = torch.randn(1, 16, 200, 352)
attn_mask = (torch.rand(1, 352, 352) > 0.7).fill_(-1000000000.0)
