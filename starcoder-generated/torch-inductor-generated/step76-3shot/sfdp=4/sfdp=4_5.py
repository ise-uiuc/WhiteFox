
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q3, K3, V3, bias):
        qk = q3 @ K3.transpose(-2, -1) / math.sqrt(q3.size(-1))
        qk = qk + bias
        attn_weight = torch.softmax(qk, -1)
        output = attn_weight @ V3
        return output
# Inputs to the model
qq = torch.randn(1, 64, 56, 56)
k2 = torch.randn(1, 64, 56, 56)
v2 = torch.randn(1, 64, 56, 56)
bias = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
