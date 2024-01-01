
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q, K, V, mask):
        qk = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))
        qk = qk + mask1
        attn_weight = torch.softmax(qk, dim=-1)
        output1 = attn_weight @ V
        qk = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))
        qk = qk + mask2
        attn_weight = torch.softmax(qk, dim=-1)
        output2 = attn_weight @ V
        return output1 + output2
# Inputs to the model
Q = torch.randn(1, 64, 56, 56)
K = torch.randn(1, 64, 56, 56)
V = torch.randn(1, 64, 56, 56)
mask1 = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
mask2 = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
