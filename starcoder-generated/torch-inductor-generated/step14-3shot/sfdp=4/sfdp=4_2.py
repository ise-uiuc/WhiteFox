
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Linear(3, 5)
        self.value = torch.nn.Linear(3, 5)
        self.query = torch.nn.Linear(3, 5)
 
    def forward(self, x1):
        k = self.key(x1)
        v = self.value(x1)
        q = self.query(x1)
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        mask4 = np.tril(np.ones((3,3))).astype('int')
        mask5 = to_tensor(mask4)
        v4 = torch.softmax(qk, dim=-1) * mask5
        v5 = v4 @ v
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
