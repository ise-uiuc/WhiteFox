
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, qk6, V3, mask):
        q_K = qk6 @ V3.transpose(-2, -1) / math.sqrt(qk6.size(-1))
        q_K = q_K + mask
        attn_weight = torch.softmax(q_K, dim=-1)
        output = attn_weight @ V3
        return output
# Inputs to the model
Q3 = torch.randn(1, 64, 56, 56)
qk5 = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
