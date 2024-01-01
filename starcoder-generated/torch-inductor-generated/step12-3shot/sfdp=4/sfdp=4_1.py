
class Model(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.query = torch.nn.Linear(hidden, hidden, bias=False)
        self.key = torch.nn.Linear(hidden, hidden, bias=False)
        self.value = torch.nn.Linear(hidden, hidden, bias=False)
 
    def forward(self, x1, x2):
        v1 = self.query(x1)
        v2 = self.key(x2)
        qk = torch.matmul(v1, v2.transpose(-2, -1))
        qk = qk.div(math.sqrt(hidden))
        qk.masked_fill_(x3, float('-inf'))
        attn_weight = torch.softmax(qk, dim=-1)
        attn = self.value(x2) @ attn_weight.transpose(-2, -1)
        return attn

# Initializing the model
m = Model(hidden=128)

# Inputs to the model
x1 = torch.randn(2, 64, 128)
x2 = torch.randn(2, 4, 128) # Key tensor should have the same shape as the query tensor
x3 = torch.randint(0, 2, (2, 64, 4)) # Attention mask
