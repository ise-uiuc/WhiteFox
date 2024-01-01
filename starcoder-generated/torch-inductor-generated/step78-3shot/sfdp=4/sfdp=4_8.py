
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q, K5, V2, mask):
        qk = Q @ K5.transpose(-2, -1) / math.sqrt(Q.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ V2
        return output
# Inputs to the model
Q9 = torch.randn(1, 64, 17, 17)
K2 = torch.randn(1, 64, 17, 17)
V = torch.randn(1, 63, 17, 17)
mask3 = (-(torch.rand(1, 17, 17) > 0.7).type(torch.FloatTensor)).fill_(0.0)
mask3 = mask3.view(1, 1, 17, 17)
mask3 = torch.cat((mask3, mask3), 1)
