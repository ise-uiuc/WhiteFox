
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q5, k, v, m4):
        Q0 = q5.transpose(1, 2)
        K = k.transpose(1, 2)
        V = v.transpose(1, 2)
        qk = Q0 @ K.transpose(-2, -1) / math.sqrt(Q0.size(-1))
        qk = qk + m4
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ V
        return output
# Inputs to the model
q = torch.randn(1, 64, 56, 56)
k = torch.randn(1, 64, 56, 56)
v = torch.randn(1, 64, 56, 56)
m = torch.rand(1, 56, 56)
mask = (m > 0.7).fill_(-1000000000.0)
