
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, query4, key3, value5, mask):
        qk = query4 @ key3.transpose(-2, -1) / math.sqrt(query4.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        outputc = attn_weight @ value5
        output2 = torch.cat((query4, outputc), 0)
        return output2
# Inputs to the model
Q3 = torch.randn(1, 64, 56, 56)
K5 = torch.randn(1, 64, 56, 56)
V10 = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
