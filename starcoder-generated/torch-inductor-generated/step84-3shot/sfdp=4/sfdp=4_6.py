
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q27, V87, K2):
        qk = Q27 @ K2.transpose(-2, -1) / math.sqrt(Q27.size(-1))
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ V87
        return output
# Inputs to the model
Q4 = torch.randn(1, 64, 56, 56)
K76 = torch.randn(1, 64, 56, 56)
V = torch.randn(1, 64, 56, 56)
