
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(32, 16, bias=False)
        self.linear2 = torch.nn.Linear(16, 8, bias=False)
        self.linear3 = torch.nn.Linear(32, 16, bias=False)
        self.linear4 = torch.nn.Linear(16, 8, bias=False)
        self.linear5 = torch.nn.Linear(32, 16, bias=False)
        self.linear6 = torch.nn.Linear(16, 8, bias=False)
        self.linear7 = torch.nn.Linear(28, 8, bias=False)

    def forward(self, q, k, v):
        q1 = self.linear1(q)
        q2 = self.linear2(q1)
        q3 = self.linear3(q)
        k1 = self.linear4(k)
        k2 = self.linear5(k1)
        k3 = self.linear6(k)
        k3.transpose_(-2, -1)
        attn_weight = torch.matmul(q2, k3)
        attn_weight = attn_weight / math.sqrt(q2.size(-1))
        mask = (torch.triu(torch.ones(q1.size(0), k1.size(0)), diagonal=1) == 1)
        attn_weight = attn_weight.masked_fill(mask, -math.inf)
        attn_weight = torch.softmax(attn_weight, dim=-1)
        output = torch.matmul(attn_weight, v)
        output.transpose_(-2, -1)
        tmp = torch.cat([output, q3], dim=-1)
        tmp.transpose_(-2, -1)
        o1 = self.linear7(tmp)
        o2 = self.linear7(output)
        return o1 + o2

# Initializing the model
m = Model()

# Weights of the model
q, k, v = torch.randn(32, 28), torch.randn(32, 32), torch.randn(32, 28) 
