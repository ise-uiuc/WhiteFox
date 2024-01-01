
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q, k, v, m):
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + m
        attention_mask = -9e15 * (1.0 - torch.erf(1.0 + k.unsqueeze(1) - q.unsqueeze(2)))
        qk = qk + attention_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v
        return output
# Inputs to the model
Q = torch.randn(1, 64, 56, 56)
K = torch.randn(1, 64, 56, 56)
V = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
