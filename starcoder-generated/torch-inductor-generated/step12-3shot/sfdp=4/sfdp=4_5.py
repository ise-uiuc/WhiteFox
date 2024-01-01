
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(4, 4)
        self.key = torch.nn.Linear(4, 4)
 
    def forward(self, q1, k2, p3=torch.tensor(1.)):
        qk = self.query(q1) @ self.key(k2).transpose(-2, -1) / math.sqrt(self.query.weight.size(-1))
        qk = qk + p3
        attn_weight = torch.softmax(qk, dim=-1)
        return attn_weight @ self.value(v3)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 4)
x2 = torch.randn(2, 4)
x3 = torch.randn(2, 4)
