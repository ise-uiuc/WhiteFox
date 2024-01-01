
class Model(torch.nn.Module):
    def forward(self, q, c, k, v2, mask):
        c = torch.sigmoid(q @ k.transpose(-2, -1))
        qk = Q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + m4
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v3
        return output
# Inputs to the model
q = torch.randn(1, 64, 56, 56)
c = torch.randn(1, 512, 8, 8)
k = torch.randn(1, 64, 56, 56)
v = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
