
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q8, k1, v8, mask):
        qk = Q8 @ k1.transpose(-2, -1) / math.sqrt(Q8.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v8
        #
        output_ = qk + qk
        #
        output_2 = output + output
        #
        output_3 = qk + output
        return output_3
# Inputs to the model
Q = torch.randn(1, 64, 56, 56)
K = torch.randn(1, 64, 56, 56)
V = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
