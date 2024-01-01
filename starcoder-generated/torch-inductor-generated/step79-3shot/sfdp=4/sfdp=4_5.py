
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, query3, Key, Value, mask):
        qk = query3 @ Key.transpose(-2, -1) / math.sqrt(query3.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ Value
        return output
# Input to the model
Q = torch.randn(1, 64, 56, 56)
K = torch.randn(1, 64, 56, 56)
V = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
