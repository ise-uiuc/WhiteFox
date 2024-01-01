
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q1, k1, v1):
        qk = q1@k1.transpose(-2, -1) * (1/math.sqrt(512))
        attn_weight = F.softmax(qk, dim=-1)
        output = (attn_weight @ v1)
        return output
# Inputs to the model
Q = torch.randn(4, 1024, 512)
K = torch.randn(4, 1024, 512)
V = torch.randn(4, 1024, 512)
mask = torch.randn(4, 1, 1,512)
mask = torch.round(torch.clamp(mask, max=10000000000.0))
