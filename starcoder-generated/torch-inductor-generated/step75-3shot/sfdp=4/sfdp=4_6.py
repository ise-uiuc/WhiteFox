
class Model(torch.nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
    def forward(self, q, k, v, mask):
        qk = torch.matmul(q, k.transpose(-2, -1))
        qk = qk / math.sqrt(q.size(-1))
        qk = qk + mask
        attn_weight = torch.nn.functional.softmax(qk, dim=-1)
        output = torch.matmul(attn_weight, v)
        return output
# Inputs to the model
Q5 = torch.randn(1, 64, 56, 56)
K6 = torch.randn(1, 64, 56, 56)
V7 = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
