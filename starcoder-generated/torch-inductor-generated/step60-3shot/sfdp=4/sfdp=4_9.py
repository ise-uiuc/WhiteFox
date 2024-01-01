
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Query31, Key12, value, mask):
        qk = Query31 @ Key12.transpose(-2, -1) / math.sqrt(Query31.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ value
        return output
# Inputs to the model
Query31 = torch.randn(1, 64, 56, 56)
Key12 = torch.randn(1, 64, 56, 56)
value = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
