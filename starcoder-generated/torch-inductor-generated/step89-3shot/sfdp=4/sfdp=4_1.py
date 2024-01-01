
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q, K, v, m):
        qk = q @ K.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + m
        attn_weight = torch.softmax(qk, dim=-1)
        output = (attn_weight @ v).transpose(1,2)
        return output
# Inputs to the model
a = torch.randn(3, 10, 64, 8, 7, 1, 1)
b = torch.randn(3, 10, 64, 8, 7, 1, 4)
c = torch.randn(3, 10, 64, 8, 7, 1, 2)
m = torch.randn(3, 10, 64, 8, 7, 1, 4)
