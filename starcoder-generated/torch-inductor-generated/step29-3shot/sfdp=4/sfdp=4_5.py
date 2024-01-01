
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q, K, V, mask):
        q = Q.transpose(0, 1)
        k = K.transpose(0, 1)
        v = V.transpose(0, 1)
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v.transpose(0, 1)
        output = output.transpose(0, 1)
        return output
# Inputs to the model
q = torch.randn(1, 56, 64)
k = torch.randn(1, 56, 64)
v = torch.randn(1, 56, 64)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
