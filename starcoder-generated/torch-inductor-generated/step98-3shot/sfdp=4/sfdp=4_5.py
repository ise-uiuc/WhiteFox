
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def _sub_func(self, Q, K, V6, mask):
        qk = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ V6
        return output
    def forward(self, Q2, K2, V6, mask2):
        output = self._sub_func(Q2, K2, V6, mask2)
        return output
# Inputs to the model
Q = torch.randn(1, 64, 56, 56)
K = torch.randn(1, 64, 56, 56)
V = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
