
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q, k, v):
        qk = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))
        attn_weight = torch.softmax(qk, -1)
        output = attn_weight @ v
        return output
# Inputs to the model
Q6 = torch.randn(1, 64, 56, 56)
K5 = torch.randn(1, 64, 56, 56)
V4 = torch.randn(1, 64, 56, 56)
