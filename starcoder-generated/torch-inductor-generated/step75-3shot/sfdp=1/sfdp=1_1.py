
class Model(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.key = torch.nn.Linear(kwargs['dimension'], kwargs['dimension'])
        self.value = torch.nn.Linear(kwargs['dimension'], kwargs['dimension'])
        self.query = torch.nn.Linear(kwargs['dimension'], kwargs['dimension'])
 
    def forward(self, x1):
        s1 = self.key(x1).transpose(-2, -1)
        v1 = self.value(x1)
        q1 = self.query(x1)
        v2 = torch.matmul(v1, s1)
        q2 = q1.unsqueeze(-2)
        q3 = q2.transpose(-2, -1)
        v3 = v2.div(0.12)
        v4 = v3.softmax(dim=-1)
        v5 = torch.nn.functional.dropout(v4, p=0.12)
        v6 = torch.matmul(v5, v1)
        v7 = torch.matmul(q2, v6)
        return v7

# Inputs to the model
x1 = torch.randn(1, 12, 32)
