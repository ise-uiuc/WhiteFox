
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q, key, value, mask):
        qk = Q @ key.transpose(-2, -1) / math.sqrt(Q.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ value
        return output
# Inputs to the model
Q = torch.randn(1,128,32,16)
key = torch.randn(1,128,32,16)
val = torch.randn(1,128,32,16)
mask = (torch.rand(1,32,16) > 0.7).fill_(-1000000000.0)
