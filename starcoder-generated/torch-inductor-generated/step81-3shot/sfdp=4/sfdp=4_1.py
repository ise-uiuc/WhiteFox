
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q1 = torch.nn.Linear(10, 10)
        self.w1 = torch.nn.Linear(10, 5)
    def forward(self, q1, w1, k):
        qk = self.q1(q1) @ self.w1(k).transpose(-2, -1) / math.sqrt(self.q1(q1).size(-1))
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ w1(v)
        return output
# Inputs to the model
q1 = torch.randn(1, 10, 224, 224)
v = torch.randn(1, 10, 224, 224)
k = torch.randn(1, 10, 224, 224)
