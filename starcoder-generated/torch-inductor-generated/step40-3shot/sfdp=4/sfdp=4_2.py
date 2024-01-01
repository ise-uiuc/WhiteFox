
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input, input2, input3, mask):
        input2 = input2.permute(1, 0, 2, 3)
        qk = input @ input2 / math.sqrt(input2.size(-1))
        qk_mask = mask.view(qk.size())
        qk = qk.masked_fill(qk_mask!= 0, qk_mask[qk_mask!= 0])
        q3 = qk.permute(1, 0, 2, 3)
        attn_weight = torch.softmax(q3, dim=-1)
        output = attn_weight @ input3
        return (q3, output)
# Inputs to the model
Q = torch.randn(1, 64, 56, 56)
K = torch.randn(1, 64, 56, 56)
V = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
