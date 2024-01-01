
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q, k, V2, mask):
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ V2
        return output
# Inputs to the model
query = torch.randn(1, 64, 784)
key = torch.randn(1, 64, 768)
value = torch.randn(1, 64, 768)
mask = (torch.rand(1, 784) > 0.7).fill_(-1000000000.0)
