
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input, mask):
        Q = input
        K = input
        V = input
        qk = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ V
        return output
# Input to model
input = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
