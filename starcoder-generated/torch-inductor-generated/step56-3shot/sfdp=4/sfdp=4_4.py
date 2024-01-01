
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, queries, keys, values, attn_mask):
        qk = queries @ keys.transpose(-2, -1) / math.sqrt(queries.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ values
        return output
# Inputs to the model
q = torch.randn(1, 64, 56, 56)
k3 = (torch.rand(1, 64, 384, 56, 56) > 0.8).int()
v3 = (torch.rand(1, 64, 384, 56, 56) > 0.8).int()
mask = (torch.rand(1, 56, 3, 56, 56) > 0.5).int()
