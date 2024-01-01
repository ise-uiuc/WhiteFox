
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, Q1, K4, V4, mask):
        qk = Q1 @ K4.transpose(-2, -1) / math.sqrt(Q1.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, -1)
        output = attn_weight @ V4
        return output
# Inputs to the model
Q6 = torch.randn(1, 768, 196)
K = torch.randn(1, 768, 196)
V = torch.randn(1, 768, 196)
mask = (torch.rand(1, 196) > 0.7).fill_(-1000000000.0)
