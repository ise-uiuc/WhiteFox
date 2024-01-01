
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q5, K, V, mask):
        qk = Q5 @ K.transpose(-2, -1) / math.sqrt(Q5.size(-1))
        qk = qk + mask
        attn = F.softmax(qk, dim=-1)
        output = attn @ V
        return output
# Inputs to the model
Q = torch.randn(1, 3, 56, 56)
K = torch.randn(1, 3, 56, 56)
V = torch.randn(1, 3, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).unsqueeze(0).unsqueeze(0).double()
mask[:, :, 14, 14] = -100000000.0
