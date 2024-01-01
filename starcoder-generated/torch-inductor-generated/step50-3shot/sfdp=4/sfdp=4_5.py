
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q):
        Q3 = nn.Conv2d(Q.shape[1], Q.shape[1], (1, 1))
        Q3.weight = torch.nn.Parameter(torch.tensor([]))
        k = torch.randn(Q.shape[1], Q.shape[1], 1, 1)
        v = Q
        mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
        qk = Q3(Q) @ k.transpose(-2, -1) / math.sqrt(Q.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v
        return output
# Inputs to the model
Q = torch.randn(1, 64, 56, 56)
