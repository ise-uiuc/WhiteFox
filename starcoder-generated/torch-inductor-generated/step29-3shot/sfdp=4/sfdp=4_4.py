
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q, k, v, mask):
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + mask
        attn_weight = torch.matmul(qk, v)
        attn_weight.masked_fill_(mask == 0.0, -1000000000.0)
        attn_weight = torch.softmax(attn_weight.float(), dim=-1)
        attn_weight = attn_weight.type_as(q)
        output = attn_weight @ v
        return output
# Inputs to the model
Q = torch.randn(1, 32, 9, 9)
K = torch.randn(1, 32, 9, 9)
V = torch.randn(1, 32, 9, 9)
mask = (torch.rand(1, 9, 9, 9) > 0.7).fill_(float('-inf'))
