
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q, K, V, mask):
        qk = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ V
        z = output @ K
        tanh = torch.tanh(z)
        z1 = tanh @ Q
        z1 = z1 + mask
        output = attn_weight @ V
        return z1, output
# Inputs to the model
Q3 = torch.randn(1, 64, 56, 56)
K4 = torch.randn(1, 64, 56, 56)
V1 = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
