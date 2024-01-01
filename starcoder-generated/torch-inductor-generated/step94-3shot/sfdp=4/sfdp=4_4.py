
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()        
    def forward(self, Q0, K3, v3, mask):
        qk = Q0 @ K3.transpose(-2, -1) / math.sqrt(Q0.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v3
        return output
# Inputs to the model
Q0 = torch.randn(1, 64, 56, 56).to(torch.float16)
K3 = torch.randn(1, 64, 56, 56).to(torch.float16)
V0 = torch.randn(1, 64, 56, 56).to(torch.float16)
mask = (torch.rand(1, 56, 56) > 0.7).to(torch.int).fill_(-1000000000)
