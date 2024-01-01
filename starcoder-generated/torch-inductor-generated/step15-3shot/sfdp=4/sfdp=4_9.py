
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(64, 64)
        self.key = torch.nn.Linear(64, 64)
        self.value = torch.nn.Linear(64, 64)
        self.attn_mask = torch.nn.Parameter(torch.tril(torch.ones(1, 96, 96)))
 
    def forward(self, x1):
        q = self.query(x1)
        k = self.key(x1)
        v = self.value(x1)
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + self.attn_mask
        weight = torch.softmax(qk, dim=-1)
        output = weight @ v
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 96, 64)
