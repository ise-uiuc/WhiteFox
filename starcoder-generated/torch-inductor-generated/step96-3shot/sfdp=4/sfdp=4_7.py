
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, query, key, value, mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 512, 196)
key = torch.randn(1, 512, 100)
value = torch.randn(1, 512, 100)
mask = (torch.rand(1, 196, 100) > 0.7).fill_(-1000000000.0)
