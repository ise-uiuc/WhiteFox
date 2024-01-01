
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q, K, V, mask):
        qk = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ V
        return output
# Inputs to the model
Q = torch.randn(1, 64, 48, 48)
K = torch.randn(1, 64, 48, 48)
V = torch.randn(1, 64, 48, 48)
mask = (torch.rand(1, 48, 48) > 0.7).fill_(float(-100000))
