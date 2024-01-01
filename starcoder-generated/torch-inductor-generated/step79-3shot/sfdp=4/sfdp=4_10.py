
class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q, K, V, mask):
        qk = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))
        qk = qk + mask
        attn_weight = nn.Softmax2d(dim=-1)(qk)
        output = attn_weight @ V
        return output
# Inputs to the model
Q2 = torch.randn(1, 56, 56, 64)
K2 = torch.randn(1, 56, 56, 64)
V2 = torch.randn(1, 56, 56, 64)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
