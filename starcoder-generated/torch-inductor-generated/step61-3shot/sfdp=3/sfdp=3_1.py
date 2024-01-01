
class Model(torch.nn.Module):
    def __init__(self, q, k, v):
        super().__init__()
        self.q = torch.nn.Linear(3, 5)
        self.k = torch.nn.Linear(5, 8)
        self.v = torch.nn.Linear(5, 8)
        self.scale_factor = 1/( math.sqrt(3)*10 )
        self.dropout_p = 0.1
 
    def forward(self, x1, x2):
        v1 = self.q(x1)
        v2 = self.k(x2)
        v3 = torch.matmul(v1, v2.transpose(-2, -1))
        v4 = v3.mul(self.scale_factor)
        v5 = torch.nn.functional.softmax(v4, dim=-1)
        v6 = torch.nn.functional.dropout(v5, p=self.dropout_p)
        v7 = torch.matmul(v6, self.v.weight.transpose(-2, -1))
        return v7

# Initializing the model
q = torch.nn.Linear(3, 5)
k = torch.nn.Linear(5, 8)
v = torch.nn.Linear(5, 8)
m = Model(q, k, v)

# Inputs to the model
x1 = torch.randn(4, 3)
x2 = torch.randn(4, 5)
