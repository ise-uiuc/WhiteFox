
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, mask1, mask):
        qk = x @ x.transpose(-2, -1) / math.sqrt(x.size(-1))
        qk = qk + mask1 + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ x
        return output
# Inputs to the model
x = torch.randn(1, 64, 56, 56)
mask1 = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000.0)
mask2 = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
