
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q, k, v5, m):
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + m
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v5
        return output
# Inputs to the model
query = torch.randn(1, 64, 56, 56)
key = torch.randn(1, 64, 56, 56)
value = torch.randn(1, 64, 56, 56)
attn_mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
