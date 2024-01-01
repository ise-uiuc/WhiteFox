
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, qw, kw, v3, qmask, kmask):
        qk = qw @ kw.transpose(-2, -1) / math.sqrt(qw.size(-1))
        qk = qk + kmask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v3
        output = output + qmask
        return output
# Inputs to the model
Q = torch.randn(1, 64, 56, 56)
K = torch.randn(1, 64, 56, 56)
V = torch.randn(1, 64, 56, 56)
qmask = (torch.rand(1, 56, 56) * -20.0).to(torch.float32)
kmask = (torch.rand(1, 56, 56) * -20.0).to(torch.float32)
