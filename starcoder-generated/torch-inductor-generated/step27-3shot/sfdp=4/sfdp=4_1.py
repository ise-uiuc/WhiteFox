
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(8, 8)
        self.key = torch.nn.Linear(8, 8)
        self.value = torch.nn.Linear(8, 8)
 
    def forward(self, x1):
        q = self.query(x1)
        k = self.key(x1)
        v = self.value(x1)
        k = k / math.sqrt(k.size(-1))
        qk = q @ k.transpose(-2, -1)
        qk = qk + attn_mask
        a = torch.softmax(qk, dim=-1)
        o = a @ v
        return o

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(12, 8)
