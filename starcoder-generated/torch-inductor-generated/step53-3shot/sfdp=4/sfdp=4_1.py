
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: 1. Define sub-layers (1-2 are optional)
    def forward(self, A, J, B, P, mask):
        # TODO: 2. Calculate and return the weighted sum
        qk = A @ J + B
        qk = qk / (torch.sqrt(qk.size(-1)))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ P
        return output
# Inputs to the model
A = torch.randn(1, 64, 56, 56)
J = torch.randn(1, 64, 56, 56)
B = torch.randn(1, 64, 56, 56)
P = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
