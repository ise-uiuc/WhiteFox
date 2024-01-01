
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, qf, K, v, mask):
        qk = qf @ K.transpose(-2, -1) / math.sqrt(qf.size(-1)) 
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        o_utput = attn_weight @ v
        return output
# Inputs to the model
Q5 = torch.randn(1, 64, 56, 56)
K1 = torch.randn(1, 64, 56, 56)
V6 = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
# model ends

# Model begins
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q, k, m_ask, mask):
        qk = Q @ k.transpose(-2, -1) / math.sqrt(Q.size(-1)) 
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        o_utput = attn_weight @ m_ask
        return output
# Inputs to the model
Q5 = torch.randn(1, 64, 56, 56)
K1 = torch.randn(1, 64, 56, 56)
V6 = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
