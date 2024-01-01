
class Model(torch.nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.query = torch.nn.modules.Linear(100, 100, bias=False)
        self.key = torch.nn.modules.Linear(100, 100, bias=False)
        self.value = torch.nn.modules.Linear(100, 100, bias=False)

    def inv_softplus(self, x):
        return 1. / torch.nn.functional.softplus(x)
    
    def forward(self, x1, x2):
        v1 = self.query(x1)
        v2 = self.key(x2)
        s1 = 1. / torch.norm(v1)
        s2 = 1. / torch.norm(v2)
        v3 = v1 * s1
        v4 = v2 * s2
        v5 = torch.matmul(v3, v4.transpose(-2, -1))
        v6 = self.inv_softplus(v5)
        v7 = torch.nn.functional.softmax(input=v6, dim=-1)
        v8 = torch.nn.functional.dropout(v7, p=0.1)
        return torch.matmul(v8, self.value(x2))


# Initializing the model
m = Model(8)

# Inputs to the model
x1 = torch.randn(1, 80, 101)
x2 = torch.randn(1, 80, 201)
