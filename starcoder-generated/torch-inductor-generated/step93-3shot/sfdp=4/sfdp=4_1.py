
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Query, Key, Val0, mask):
        qk = Query @ Key.transpose(-2, -1) / math.sqrt(Query.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ Val0
        return output
# Inputs to the model
query = torch.randn(1, 64, 56, 56)
key = torch.randn(1, 64, 56, 56)
val = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
