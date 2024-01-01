
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
Q = torch.randn(1, 8, 16, 32)
K = torch.randn(1, 8, 32, 16)
V = torch.randn(1, 8, 32, 16)
mask = torch.randn(8, 16, 32).ge(0).float().fill_(-10000.0)
mask = mask.unsqueeze(0)
