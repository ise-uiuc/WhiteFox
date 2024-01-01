
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q0, K0, V):
        qk = Q0 @ K0.transpose(-2, -1) / math.sqrt(Q0.size(-1))
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ V
        return output
# Inputs to the model
Q = torch.randn(1, 64, 56, 56)
K = torch.randn(1, 64, 56, 56)
V = torch.randn(1, 64, 56, 56)
mask = torch.Tensor()
