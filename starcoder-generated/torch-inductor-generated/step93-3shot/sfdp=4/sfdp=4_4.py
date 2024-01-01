
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q1, k, v, mask):
        qk = q1 @ k.transpose(-2, -1) / math.sqrt(q1.size(-1))
        qk = torch.add(qk, mask)
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v
        return output
# Inputs to the model
Q4 = torch.randn(1, 256, 8, 8)
K5 = torch.randn(1, 256, 8, 8)
V6 = torch.randn(1, 256, 8, 8)
mask8 = torch.empty([1, 8, 8], dtype = Q4.dtype, device=Q4.device).uniform_(0, 1) > 0.7
