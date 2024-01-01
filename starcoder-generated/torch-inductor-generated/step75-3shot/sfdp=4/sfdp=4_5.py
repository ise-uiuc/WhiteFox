
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q, k, v, mask):
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, -1)
        output = attn_weight @ v
        return output
# Inputs to the model
Q5 = torch.randn(1, 64, 28, 28)
K6 = torch.randn(1, 64, 28, 28)
V6 = torch.randn(1, 64, 28, 28)
mask = (torch.rand(1, 28, 28) > 0.7).fill_(-1000000000.0)
