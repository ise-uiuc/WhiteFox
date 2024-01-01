
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q2, KKKKKKKK2, VVVVVVVVVVVVVVVVVV3, mask):
        qk = <EMAIL>.transpose(-2, -1) / math.sqrt(Q2.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ VVVVVVVVVVVVVVVVVV3
        return output
# Inputs to the model
Q3 = torch.randn(1, 64, 56, 56)
KKKKKKKKKKKKKKKKKKK3 = torch.randn(1, 64, 56, 56)
VVVVVVVVVVVVVVVVVVVVV3 = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
