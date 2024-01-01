
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q, K7, V1, mask):
        qk = Q @ K7.transpose(-2, -1) / math.sqrt(Q.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ V1
        return output
# Inputs to the model
Q = torch.randn(1, 64, 56, 56)
K4 = torch.randn(1, 64, 56, 56)
V1 = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
