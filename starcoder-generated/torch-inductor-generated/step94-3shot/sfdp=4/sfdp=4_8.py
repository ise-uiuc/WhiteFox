
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, query, keys, values, mask):
        qk = query @ keys.transpose(-2, -1) / math.sqrt(query.size(-1))
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ values
        return output
# Inputs to the model
Q9 = torch.randn(1, 64, 46, 46)
K3 = torch.randn(1, 64, 46, 46)
V2 = torch.randn(1, 64, 46, 46)
mask = torch.randn(1, 46, 3, 46) > 0.7
