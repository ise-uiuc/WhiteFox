
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
Q = torch.rand(1, 8,8,8)
K = torch.rand(1, 8,8,8)
V = torch.rand(1, 8,8,8 )
mask = (torch.rand(1, 8, 8) > 0.7).fill_(float(-100000))
