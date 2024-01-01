
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = torch.nn.Linear(25, 25)
        self.wk = torch.nn.Linear(25, 25)

        self.qw = torch.nn.Parameter(torch.zeros(1, 1, 25))
        self.kw = torch.nn.Parameter(torch.zeros(1, 1, 25))

    def forward(self, q, k, v, attn_mask):
        q = self.wq(q)
        k = self.wk(k)
        qw = torch.matmul(self.qw, q.unsqueeze(2)).squeeze(2)
        kw = torch.matmul(self.kw, k.unsqueeze(2)).squeeze(2)
        qk = torch.div(qw, 25) @ torch.div(kw, 25).T
        qk = qk + attn_mask
        attn_weight = F.softmax(qk, 1)
        v = self.wv(v)
        output = torch.matmul(attn_weight, v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 5, 25)
k = torch.randn(1, 5, 25)
v = torch.randn(1, 5, 25)
attn_mask = -float('inf') * torch.ones(k.shape[0], k.shape[1], requires_grad=False)
