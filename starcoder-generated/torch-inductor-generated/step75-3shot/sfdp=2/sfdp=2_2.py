
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        dim = 10
        self.key = torch.nn.Linear(dim, dim)
        self.query = torch.nn.Linear(dim, dim)
        self.value = torch.nn.Linear(dim, dim)
 
    def forward(self, x1, x2):
        v1 = self.query(x1)
        v2 = self.key(x2)
        v3 = v1.bmm(v2.transpose(-2, -1))
        v4 = 1 / math.sqrt(10)
        v5 = v3.div(v4)
        v6 = torch.nn.functional.softmax(v5, dim=-1)
        v7 = torch.nn.functional.dropout(v6, p=0.5)
        v8 = self.value(v7)
        v9 = v8.bmm(x1.transpose(-2, -1))
        return v9

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10, 10)
x2 = torch.randn(10, 10)
