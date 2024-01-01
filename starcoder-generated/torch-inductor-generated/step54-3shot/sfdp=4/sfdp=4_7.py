
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input, mask):
        qk = input @ input.transpose(-2, -1) / math.sqrt(input.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ input
        return output
model = Model()
# Inputs to the model
Q = torch.randn(1, 64, 56, 56)
K = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
