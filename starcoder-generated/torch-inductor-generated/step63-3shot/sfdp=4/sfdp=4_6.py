
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q, K, V1, mask):
        qk = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        outpu = attn_weight @ V1
        return outpu
# Inputs to the model
Q = torch.randn(1, 1024, 192)
K = torch.randn(1, 256, 256)
V = torch.randn(1, 1024, 256)
mask = (torch.rand(1, 192, 192) > 0.7).fill_(-1000000000.0)
