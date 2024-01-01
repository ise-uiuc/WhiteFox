
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, query, key, value, m6):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + m6
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ value
        return output
# Inputs to the model
Q = torch.randn(1, 2304, 7, 7)
Key = torch.randn(1, 2304, 7, 7)
V = torch.randn(1, 2304, 7, 7)
mask = (torch.rand(1, 7, 7) > 0.7).fill_(-1000000000.0)
