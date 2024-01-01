
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, query3, key, value1, mask,):
        query_key = query3 @ key.transpose(-2, -1) / math.sqrt(query3.size(-1))
        query_key = query_key + mask
        attn_weight = torch.softmax(query_key, dim=-1)
        output = attn_weight @ value1
        return output
# Inputs to the model
Q13 = torch.randn(1, 64, 56, 56)
K = torch.randn(1, 64, 56, 56)
V = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
