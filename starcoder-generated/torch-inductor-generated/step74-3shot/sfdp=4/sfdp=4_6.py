
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q, K, V1, mask):
        qK = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))
        qK = qK + mask
        attn_weight = torch.softmax(qK, dim=-1)
        output = attn_weight @ V1
        return output
# Inputs to the model
Q5 = torch.randn(1, 64, 56, 56)
K6 = torch.randn(1, 64, 56, 56)
V2 = torch.randn(1, 64, 56, 56)
mask3 = (torch.rand(1, 56, 56) > 0.7).fill_(-10000000000000.0)
