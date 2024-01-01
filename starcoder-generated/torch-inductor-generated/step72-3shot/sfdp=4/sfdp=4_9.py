
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(56, 42)
        self.linear2 = torch.nn.Linear(42,56)
    def forward(self, q, k, v, m):
        qk = q @ k.transpose(-2, -1)
        qk = qk / math.sqrt(q.size(-1))
        qk = qk + m
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v

        qk2 = self.linear1(q) @ k.transpose(-2, -1)
        qk2 = qk2 / math.sqrt(qk2.size(-1))
        qk2 = qk2 + m
        attn_weight2 = torch.softmax(qk2, dim=-1)
        output2 = attn_weight2 @ v

        qk3 = self.linear2(q) @ self.linear2(k).transpose(-2, -1)
        qk3 = qk3 / math.sqrt(qk3.size(-1))
        qk3 = qk3 + m
        attn_weight3 = torch.softmax(qk3, dim=-1)
        output3 = attn_weight3 @ v

        return output,output2,output3
# Inputs to the model
Q = torch.randn(1, 56)
K = torch.randn(1, 56)
V = torch.randn(1, 56)
mask = (torch.rand(1, 56) > 0.7).fill_(-1000000000.0)
