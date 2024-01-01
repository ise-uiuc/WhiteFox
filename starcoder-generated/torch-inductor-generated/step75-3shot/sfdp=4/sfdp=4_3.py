
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q, k, v, m1, m2):
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + m1
        attn_weight1 = torch.softmax(qk, dim=-1)
        output1 = attn_weight1 @ v
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + m2
        attn_weight2 = torch.softmax(qk, dim=-1)
        output2 = attn_weight2 @ v
        return output1, output2
# Inputs to the model
Q3 = torch.randn(1, 64, 56, 56)
K = torch.randn(1, 64, 56, 56)
V = torch.randn(1, 64, 56, 56)
m1 = torch.rand(1, 56, 56)
m2 = torch.rand(1, 56, 56)
