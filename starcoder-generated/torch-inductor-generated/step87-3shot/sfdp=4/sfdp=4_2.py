
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q, k, v, mask):
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 1, 3)
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v
        output = output.permute(0, 2, 1, 3)
        return output
# Inputs to the model
Q = torch.randn(1, 32, 56, 56)
k = torch.randn(1, 32, 56, 56)
v = torch.randn(1, 32, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
