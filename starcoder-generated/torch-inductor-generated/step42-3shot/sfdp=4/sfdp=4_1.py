
class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q3, K, V, mask):
        qk = torch.matmul(Q3, K.transpose(-2, -1))
        qk = qk / math.sqrt(Q3.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = torch.matmul(attn_weight, V)
        return output
# Inputs to the model
Q = torch.randn(1, 64, 56, 56)
K = torch.randn(1, 64, 56, 56)
V = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
