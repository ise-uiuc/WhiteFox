
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(8, 1, 7, 1))
 
    def forward(self, x2):
        v7 = self.query.matmul(x2).squeeze(-1) / math.sqrt(7)
        v8 = v7 + 1
        v9 = F.softmax(v8, dim=-1)
        v10 = x2.matmul(v9.unsqueeze(-1)).squeeze(-1)
        return v10

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(8, 1, 7, 1)
