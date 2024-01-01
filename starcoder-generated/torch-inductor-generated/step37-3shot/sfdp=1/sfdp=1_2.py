
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_q = torch.nn.Linear(3072, 4096)
        self.proj_k = torch.nn.Linear(3072, 4096)
        self.proj_v = torch.nn.Linear(3072, 4096)
 
    def forward(self, x1, x2):
        v1 = self.proj_q(x1)
        v2 = self.proj_k(x2)
        v3 = self.proj_v(x2)
        v4 = torch.matmul(v1, v2.transpose(-2, -1))
        v5 = v4.div(2048.0)
        v6 = v5.softmax(dim=-1)
        v7 = torch.nn.functional.dropout(v6, 0.1)
        v8 = torch.matmul(v7, v3)
        return v8

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3072)
x2 = torch.randn(1, 3072)
