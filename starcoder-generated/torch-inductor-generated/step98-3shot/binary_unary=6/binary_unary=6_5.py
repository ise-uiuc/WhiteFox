
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(30, 1, bias=False)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - 0.75
        v3 = v1 - 0.2857142857142857
        v4 = v1 + 0.0357142857142857
        v5 = -v1 - 0.7142857142857143
        v6 = v2 * v3 
        v7 = v2 + v3
        v8 = torch.cosine_similarity(v4, v5)
        v9 = torch.addcmul(v6, v4, v5)
        v10 = torch.addmm(v7, v4, v5)
        v11 = torch.linalg.cholesky(v8)
        return v9 + v10 + v11

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 30)
