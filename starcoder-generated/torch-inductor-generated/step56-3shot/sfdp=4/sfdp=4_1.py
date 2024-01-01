
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q5, K, V, mask):
        qk = q5 @ K.transpose(-2, -1) / math.sqrt(q5.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ V
        return output
# Inputs to the model
Q = torch.randn(1, 64, 56, 56)
k_1 = torch.randn(1, 64, 56, 56)
v = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
