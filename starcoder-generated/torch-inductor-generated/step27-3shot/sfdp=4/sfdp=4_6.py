
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(75, 194, bias=False)
        self.key = torch.nn.Linear(75, 194, bias=False)
        self.value = torch.nn.Linear(75, 75, bias=False)
 
    def forward(self, x1, x2, x3):
        q = self.query(x1)
        k = self.key(x2)
        v = self.value(x3)
        qk = q @ k.T / np.sqrt(q.size(-1))
        m = torch.zeros((q.size(0), 194, qk.size(-1)), dtype=qk.dtype)
        vmask = v >= 0
        m[vmask] = qk[vmask]
        return m

# Initializing the model
m = Model() # Model with different bias of q, k and v
 
# Inputs to the model
x1 = torch.randn(1, 194, 75) # input tensor (query)
x2 = torch.randn(1, 194, 75) # input tensor (key)
x3 = torch.randn(1, 194, 75) # input tensor (value)
