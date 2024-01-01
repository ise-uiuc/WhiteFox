
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q20, K2, VVV6, mask):
        qk = Q20 @ K2.transpose(-2, -1) / math.sqrt(Q20.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ VVV6
        return output
# Inputs to the model
Q14 = torch.randn(1, 64, 56, 56)
K7 = torch.randn(1, 64, 56, 56)
VVV1 = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
model = Model()
model, _ = torch.jit.get_trace_graph(model, (Q14, K7, VVV1, mask))
print(model) # This generated a different forward
