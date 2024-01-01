
class Model(torch.nn.Module):
    def __init__(self, nHeads, queryDim, keyDim, valueDim):
        super().__init__()
        self.nHeads = nHeads
        self.query = torch.nn.Linear(queryDim, nHeads * keyDim, bias=False)
        self.key = torch.nn.Linear(keyDim, nHeads * keyDim, bias=False)
        self.value = torch.nn.Linear(valueDim, nHeads * valueDim, bias=False)
        self.attn_mask = torch.randn(3, keyDim, queryDim) * 1E-5
 
    def forward(self, x1, x2):
        qk = self.query(x1) @ self.key(x2).transpose(-2, -1) / math.sqrt(x1.size(-1))
        qk = qk + self.attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ self.value(x2)
        return output

# Initializing the model
m = Model(nHeads=3, queryDim=24, keyDim=24, valueDim=24)

# Inputs to the model
x1 = torch.randn(4, 24)
x2 = torch.randn(2, 3, 24)
