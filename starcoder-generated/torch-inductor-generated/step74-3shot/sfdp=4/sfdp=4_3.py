
class Model(torch.nn.Module):
    def __init__(self, a, b, c):
        super().__init__()
    def forward(self, Q, K, V1, mask):
        qK = Q @ K.transpose(-2, -1) / math.sqrt(K.size(-1))
        qK = qK + mask
        attn_weight = torch.softmax(qK, dim=-1)
        output = attn_weight @ V1
        return output
# Inputs to the model
q = torch.randn(1, 64, 56, 56)
k = torch.randn(1, 64, 56, 56)
v = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
