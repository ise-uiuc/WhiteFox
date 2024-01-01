
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q, k, v, mask):
        Q = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        Q = + mask
        q = torch.softmax(Q, dim=-1)
        output = attn_weight @ V
        return output
# Inputs to the model
Q = torch.randn(1, 64, 56, 56)
K = torch.randn(1, 64, 56, 56)
V = torch.randn(1, 64, 56, 56)
mask = (torch.rand(Q.size() > 0.7)).fill_(-1000000000.0)
