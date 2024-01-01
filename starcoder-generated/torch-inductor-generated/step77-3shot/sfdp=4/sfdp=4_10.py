
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input, output):
        Q = input
        K = output
        qk = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ Q
        return output
# Inputs to the model
Q = torch.randn(1, 64, 56, 56)
K = torch.randn(1, 64, 56, 56)
