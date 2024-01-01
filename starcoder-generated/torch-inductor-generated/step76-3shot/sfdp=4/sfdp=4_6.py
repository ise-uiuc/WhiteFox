
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, query, key, val, mask_val):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + mask_val
        attn_weight = torch.softmax(qk, -1)
        output = attn_weight @ val
        return output
# Inputs to the model
query = torch.randn(1, 64, 56, 56)
key = torch.randn(1, 64, 56, 56)
val = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
